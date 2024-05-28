import depthai
import pathlib
import numpy as np

ip = "10.0.0.13"
camera = depthai.CameraBoardSocket.RIGHT


results_path = pathlib.Path(
    "~/Documents/oakd_calibration/oakd_bottom-right/results"
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

    calibration.setCameraIntrinsics(camera, mtx, w, h)
    calibration.setDistortionCoefficients(camera, dist)

    print("flashing...")
    success = device.flashCalibration(calibration)
    print(f"success: {success}")

    pass


if __name__ == "__main__":
    main()