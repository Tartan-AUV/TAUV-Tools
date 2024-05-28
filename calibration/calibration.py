import numpy as np
import cv2 as cv
import glob
import pathlib

# number of internal corners. n_cols, n_rows.
grid_size = (8, 6)
# size of grid squares, in m
grid_spacing = 0.09

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
objp[:, :2] = grid_spacing * np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

dataset_dir = pathlib.Path(
    "~/Documents/oakd_calibration/oakd_bottom-right"
).expanduser()

images = glob.glob(str(dataset_dir / "in" / "*.png"))

good_images = []

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (8, 6), None)

    if not ret:
        continue

    good_images.append(fname)

    objpoints.append(objp)

    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    imgpoints.append(corners2)

    # Draw and display the corners
    cv.drawChessboardCorners(img, (8, 6), corners2, ret)
    cv.imshow('img', img)
    cv.waitKey(100)

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None, flags=cv.CALIB_RATIONAL_MODEL)

res = np.array([img.shape[0], img.shape[1]], dtype=np.int32)

print(f"mtx: {mtx}")
print(f"rvecs: {rvecs}")
print(f"tvecs: {tvecs}")

np.savetxt(str(dataset_dir / "results" / "mtx.txt"), mtx)
np.savetxt(str(dataset_dir / "results" / "dist.txt"), dist)
np.savetxt(str(dataset_dir / "results" / "res.txt"), res)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error

print("total error: {}".format(mean_error / len(objpoints)))

for i, fname in enumerate(good_images):
    name = pathlib.Path(fname).name

    img = cv.imread(fname)

    cv.drawFrameAxes(img, mtx, dist, rvecs[i], tvecs[i], 0.09, 3)

    cv.imshow('img', img)
    cv.waitKey(100)

    cv.imwrite(str(dataset_dir / "out" / name), img)
