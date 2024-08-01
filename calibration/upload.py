import depthai
import pathlib
import numpy as np

'''
USE WITH CAUTION. This does not play nicely with stereo depth.
'''

ip = "10.0.0.12"
camera = depthai.CameraBoardSocket.LEFT


results_path = pathlib.Path(
    "~/Documents/oakd_calibration/oakd_bottom-left/results"
).expanduser()


def main():

    mtx_path = results_path / "mtx.txt"
    dist_path = results_path / "dist.txt"
    res_path = results_path / "res.txt"

    mtx = np.loadtxt(mtx_path)
    dist = np.loadtxt(dist_path)
    res = np.loadtxt(res_path)

    pipeline = depthai.Pipeline()

    print("connecting...")

    device = None
    while device is None:
        try:
            device_info = depthai.DeviceInfo(ip)

            device = depthai.Device(pipeline, device_info)
        except Exception as e:
            print(f'OAKD device error: {e}')

    print("connected!")

    calibration = device.readCalibration()

    print(f"mtx: {mtx}")
    print(f"dist: {dist}")
    print(f"res: {res}")

    (w, h) = (int(res[1]), int(res[0]))

    modified_mtx = np.array(calibration.getCameraIntrinsics(camera, w, h))
    modified_mtx[0, 0] *= 1.33
    modified_mtx[1, 1] *= 1.33

    calibration.setCameraIntrinsics(camera, modified_mtx, w, h)
    # calibration.setDistortionCoefficients(camera, dist)

    print("flashing...")
    success = device.flashCalibration(calibration)
    print(f"success: {success}")

    pass


if __name__ == "__main__":
    main()