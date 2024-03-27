# python detect.py --source ./inference/images/ --weights best.pt --conf 0.4

import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import math

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, plot_point, plot_all_point, plot_cap_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized

import heapq
import csv
import pandas as pd

def detect(save_img=False):
    csv_save_dir, out, out_cap, output_cap_resize, source, weights, view_img, save_txt, imgsz = \
        opt.csv_save_dir, opt.output, opt.output_cap, opt.output_cap_resize, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

    # Initialize
    set_logging()
    device = select_device(opt.device)

    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    # if os.path.exists(out_cap):
    #     shutil.rmtree(out_cap)
    # os.makedirs(out_cap)
    # if os.path.exists(output_cap_resize):
    #     shutil.rmtree(output_cap_resize)
    # os.makedirs(output_cap_resize)
    
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)
    print('dataset:', dataset)
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    img_x=[]
    img_y=[]
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    with open('./output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
    
            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]
    
            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = time_synchronized()
    
            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)
    
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s
                    
                x_list=[]
                y_list=[]
                save_path = str(Path(out) / Path(p).name)
                save_cap_path = str(Path(out_cap) / Path(p).name)
                save_cap_resize_path = str(Path(output_cap_resize) / Path(p).name)
                if not os.path.exists(csv_save_dir):
                    os.makedirs(csv_save_dir)
                print('\nPath:', path.rfind('images'))
                print('\nPath:', path[path.rfind('images')+7:-4])
                file_name = path[path.rfind('images')+7:-4]
                
                txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
    
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string
                    # print('\nnumber:', int(n))
                    # Write results
                    cor = []
                    for *xyxy, conf, cls in reversed(det):
                        # print('xyxy:', xyxy)
                        cor.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), abs(int(xyxy[1])-int(xyxy[3]))])
                        # print('\ncor:', cor)
                        
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
    
                        if save_img or view_img:  # Add bbox to image
                            
                            label = '%s %.2f' % (names[int(cls)], conf)
                            x_list+=[int(xyxy[0])]
                            y_list+=[int(xyxy[1])]
                            
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                            # center = (int(xyxy[0])+int(((int(xyxy[2])-int(xyxy[0])))/2), int(xyxy[1])+int((int(xyxy[3])-int(xyxy[1]))/2))
                            # writer.writerow([Path(p).name, str(center)])
                            
                    cor_data = pd.DataFrame(cor, columns=['X_1', 'Y_1', 'X_2', 'Y_2', 'height'])
                    cor_data.to_csv(os.path.join(csv_save_dir,(file_name+'.csv')), index=False)
                    
            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    print('save_path:',save_path)
                    cv2.imwrite(save_path, im0) 
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform.system() == 'Darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/exp0_data/drone/weights/drone_best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/drone/val_images/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/drone', help='output folder')
    parser.add_argument('--csv_save_dir', type=str, default='inference/drone_csv', help='csv output folder')
    parser.add_argument('--output-cap', type=str, default='inference/capture', help='output capture folder')# output folder
    parser.add_argument('--output-cap-resize', type=str, default='inference/cap_resize', help='output capture-resize folder')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            print('else')
            detect()
