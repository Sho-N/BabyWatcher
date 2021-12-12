import jetson.inference
import jetson.utils

import argparse
import sys


def parse_opt():
    parser = argparse.ArgumentParser(
        description="Locate objects in a live camera stream using an object detection DNN.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=jetson.inference.detectNet.Usage() +
               jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

    parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
    parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
    parser.add_argument("--network", type=str, default="ssd-mobilenet-v2",
                        help="pre-trained model to load (see below for options)")
    parser.add_argument("--model", type=str, default="", help="")
    parser.add_argument("--dnn-model", type=str, default="", help="")
    parser.add_argument("--labels", type=str, default="", help="")
    parser.add_argument("--input-blob", type=str, default="", help="")
    parser.add_argument("--output-cvg", type=str, default="", help="")
    parser.add_argument("--output-bbox", type=str, default="", help="")
    parser.add_argument("--overlay", type=str, default="box,labels,conf",
                        help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
    parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use")

    is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

    try:
        opt = parser.parse_known_args()[0]
        opt['is_headless'] = is_headless
    except Exception as e:
        print(e)
        parser.print_help()
        sys.exit(0)

    return opt


def main(opt):
    # load the object detection network
    net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)

    # create video sources & outputs
    video_in = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
    video_out = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv + opt.is_headless)

    while video_out.IsStreaming():
        img = video_in.Capture()
        detections = net.Detect(img)
        for detection in detections:
            video_out.Render(img)
            video_out.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))


if __name__ == '__main__':
    option = parse_opt()
    main(option)
