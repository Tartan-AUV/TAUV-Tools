import depthai
import pathlib
import numpy as np

'''
USE WITH CAUTION. This does not play nicely with stereo depth.
'''

ip = "10.0.0.13"

cameras = [
    depthai.CameraBoardSocket.LEFT,
    depthai.CameraBoardSocket.RIGHT,
    depthai.CameraBoardSocket.RGB,
]

ior_factor = 1.33


def main():
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

    for camera in cameras:
        w, h = 1920, 1080

        modified_mtx = np.array(calibration.getCameraIntrinsics(camera, w, h))
        modified_mtx[0, 0] *= ior_factor
        modified_mtx[1, 1] *= ior_factor

        calibration.setCameraIntrinsics(camera, modified_mtx, w, h)

    print("flashing...")
    success = device.flashCalibration(calibration)
    print(f"success: {success}")


if __name__ == "__main__":
    main()
