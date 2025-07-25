import sys
from pathlib import Path
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

from PIL import Image
import torchvision.transforms as transforms

# Add YOLOv5 to system path
yolov5_path = Path(__file__).resolve().parent / "yolov5"
if str(yolov5_path) not in sys.path:
    sys.path.insert(0, str(yolov5_path))

from utils.dataloaders import LoadImages, LoadStreams
from utils.general import (
    check_img_size, non_max_suppression,
    scale_boxes, xyxy2xywh
)
from utils.torch_utils import select_device, time_sync


from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from counter.draw_counter import draw_up_down_counter
import argparse
import platform
import shutil
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import cv2
import os
import math
from PIL import Image
import matplotlib.pyplot as plt
from pylab import *
from matplotlib.pyplot import ginput, ion, ioff, imshow
from numpy import array
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, './yolov5')


# COCO class indices for vehicles
VEHICLE_CLASSES = {
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}

def find_accidents(outputs, prev_tracks, frame_count_threshold=5, t=10):
    """
    Detects collisions based on DeepSORT tracking IDs and advanced overlap patterns
    """
    is_accident_happen = []
    crash_index = []
    updated_tracks = {}

    # Update individual track history
    for box in outputs:
        x1, y1, x2, y2, track_id, _ = box
        updated_tracks[track_id] = {
            "bbox": [x1, y1, x2, y2],
            "frames": prev_tracks.get(track_id, {}).get("frames", 0) + 1
        }

    # Check for collision using detailed spatial patterns
    for i in range(len(outputs)):
        A_xmin, A_ymin, A_xmax, A_ymax, track_id1, class1 = outputs[i]

        for j in range(i + 1, len(outputs)):
            B_xmin, B_ymin, B_xmax, B_ymax, track_id2, class2 = outputs[j]

            if track_id1 == track_id2:
                continue

            collided = False

             # Pattern 01: A is mostly inside B
            if (B_xmin < A_xmin + t < B_xmax - t and A_xmax > B_xmax - t) and \
            (B_ymin < A_ymin + t < B_ymax - t and A_ymax > B_ymax - t):
                collided = True

            # Pattern 02: A enters from top-left into B
            elif (B_xmin < A_xmin + t < B_xmax - t and A_xmax > B_xmax - t) and \
                (A_ymin < B_ymin < B_ymax and A_ymax < B_ymax):
                collided = True

            # Pattern 03: A partially overlaps top of B
            elif (B_xmin < A_xmin + t < B_xmax - t and A_xmax > B_xmax - t) and \
                (A_ymin < B_ymin + t < A_ymax - t and B_ymax > A_ymax - t):
                collided = True

            # Pattern 04: A contains B horizontally and overlaps top edge
            elif (A_xmin < B_xmin < B_xmax and A_xmax > B_xmax) and \
                (B_ymin < A_ymin + t < B_ymax - t and A_ymax > B_ymax - t):
                collided = True

            # Pattern 05: B partially enters A from left side
            elif (A_xmin < B_xmin + t < A_xmax - t and B_xmax > A_xmax - t) and \
                (B_ymin < A_ymin + t < B_ymax - t and A_ymax > B_ymax - t):
                collided = True

            # Pattern 06: B enters A from left with vertical containment
            elif (A_xmin < B_xmin + t < A_xmax - t and B_xmax > A_xmax - t) and \
                (A_ymin < B_ymin < B_ymax and A_ymax > B_ymax):
                collided = True

            # Pattern 07: B overlaps A horizontally with partial vertical inclusion
            elif (A_xmin < B_xmin + t < A_xmax - t and B_xmax > A_xmax - t) and \
                (A_ymin < B_ymin + t < A_ymax - t and B_ymax > A_ymax - t):
                collided = True

            # Pattern 08: A contains B horizontally, B overlaps lower part of A
            elif (A_xmin < B_xmin < B_xmax and A_xmax > B_xmax) and \
                (A_ymin < B_ymin + t < A_ymax - t and B_ymax > A_ymax - t):
                collided = True

            # Pattern 09: B contains A vertically and overlaps right part
            elif (A_xmin < B_xmin < A_xmax and B_xmax > A_xmax) and \
                (B_ymin < A_ymin < A_ymax and B_ymax > A_ymax):
                collided = True

            # Pattern 10: B wraps horizontally around A
            elif (B_xmin < A_xmin < B_xmax and A_xmax > B_xmax) and \
                (B_ymin < A_ymin < A_ymax and B_ymax > A_ymax):
                collided = True

            # Pattern 11: B enters from left and contains A vertically
            elif (B_xmin < A_xmin < A_xmax and B_xmax > A_xmax) and \
                (A_ymin < B_ymin < A_ymax and B_ymax > A_ymax):
                collided = True

            # Pattern 12: B overlaps vertically and enters A from top
            elif (B_xmin < A_xmin < A_xmax and B_xmax > A_xmax) and \
                (B_ymin < A_ymin < B_ymax and A_ymax > B_ymax):
                collided = True

            # Count and update collision if a pattern matched
            if collided:
                key = tuple(sorted((track_id1, track_id2)))
                collision_frames = prev_tracks.get(key, 0) + 1

                if collision_frames >= frame_count_threshold:
                    is_accident_happen.append((class1, class2))
                    crash_index.append((i, j))

                updated_tracks[key] = collision_frames

    return is_accident_happen, crash_index, updated_tracks



def count_vehicles_and_check_restrictions(im0, outputs, line_pixel, counter_recording,
                                     up_counter, down_counter, class_names):

    '''
    Tracks and count vehicle movement (up/down) lane for traffic flow and detects unauthorized entry into restricted zones
    '''      
   
    if isinstance(line_pixel, list):
        line_pixel = line_pixel[0]

    # Get only relevant class IDs (e.g. car, bus, truck)
    vehicle_class_ids = list(class_names.keys())

    # Calculate center of each bounding box for better approximation where is the vehicle
    box_centers = []
    for each_box in outputs:
        # [x1, y1, x2, y2, track_id, class_id]
        x_center = (each_box[0] + each_box[2]) / 2
        y_center = (each_box[1] + each_box[3]) / 2
        track_id = int(each_box[4])
        class_id = int(each_box[5])

        # Filter out non-vehicle classes
        if class_id not in vehicle_class_ids:
            continue

        # pixel_width: width of the bounding box
        pixel_width = each_box[2] - each_box[0]
        box_centers.append([x_center, y_center, track_id, class_id, pixel_width])

    # Count vehicles going up or down
    for center in box_centers:
        x_center, y_center, track_id, class_id, pixel_width = center

        # Skip if this vehicle has already been counted 
        if track_id in counter_recording:
            continue

        if y_center >= line_pixel:
            down_counter[class_id] += 1
            counter_recording.append(track_id)
        elif y_center < line_pixel:
            up_counter[class_id] += 1
            counter_recording.append(track_id)


    # Load car line (restricted area boundaries)
    # 4 lines equation: y = kx + b
    k, b, x1, y1, x2, y2 = [], [], [], [], [], []
    with open('car_line.txt', 'r') as f:
        for line in f:
            line = eval(line.strip('\n'))
            k.append(line['k'])
            b.append(line['b'])
            x1.append(line['x1'])
            y1.append(line['y1'])
            x2.append(line['x2'])
            y2.append(line['y2'])

    for i in range(len(x1)):
        cv2.line(im0, (int(x1[i]), int(y1[i])), (int(x2[i]), int(y2[i])), (0, 0, 255), 2, cv2.LINE_AA)

    # Detect and annotate illegal entries in restricted zone
    for center in box_centers:
        x_center, y_center, track_id, class_id, pixel_width = center

        # Check if inside restricted rectangle defined by 4 lines
        inside_restricted_area = (
            k[0] * x_center + b[0] <= y_center and
            k[1] * x_center + b[1] >= y_center and
            k[2] * x_center + b[2] >= y_center and
            k[3] * x_center + b[3] <= y_center
        )

        # Show alert for vehicle illegal entry 
        if inside_restricted_area:
            class_name = class_names.get(class_id, f'id_{class_id}')
            cv2.putText(im0, f'{class_name} entering illegally!',
                        (int(x_center - 140), int(y_center)),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, [0, 0, 255], 2)

    return counter_recording, up_counter, down_counter, box_centers, im0
    
import math
import cv2

def estimate_speed(locations, fps, width, frame, speed_limit=30):
    """
    Estimate vehicle speed and draw a warning on the frame if speed exceeds threshold.
    
    Parameters:
        locations: [prev_frame_locations, current_frame_locations]
        fps: frames per second of the video
        width: dictionary of real-world average widths for each class
        frame: current video frame to draw warnings
        speed_limit: km/h threshold to flag overspeeding
    Returns:
        speed: list of [speed, vehicle_id, overspeed_flag]
        frame: modified frame with text drawn (if any)
    """
    present_IDs = []
    prev_IDs = []
    work_IDs = []
    work_IDs_index = []
    work_IDs_prev_index = []

    work_locations = []
    work_prev_locations = []

    for i in range(len(locations[1])):
        present_IDs.append(locations[1][i][2])
    for i in range(len(locations[0])):
        prev_IDs.append(locations[0][i][2])

    for m, n in enumerate(present_IDs):
        if n in prev_IDs:
            work_IDs.append(n)
            work_IDs_index.append(m)
    for x in work_IDs_index:
        work_locations.append(locations[1][x])
    for y, z in enumerate(prev_IDs):
        if z in work_IDs:
            work_IDs_prev_index.append(y)
    for x in work_IDs_prev_index:
        work_prev_locations.append(locations[0][x])

    speed = []

    for i in range(len(work_IDs)):
        x1, y1 = work_locations[i][0], work_locations[i][1]
        x0, y0 = work_prev_locations[i][0], work_prev_locations[i][1]

        dx = x1 - x0
        dy = y1 - y0
        pixel_distance = math.sqrt(dx ** 2 + dy ** 2)

        vehicle_class = work_locations[i][3]
        pixel_width = work_locations[i][4]
        real_width_meters = width[vehicle_class]

        meters_per_pixel = real_width_meters / pixel_width
        real_distance_m = pixel_distance * meters_per_pixel

        time_seconds = 5 / fps
        speed_m_per_s = real_distance_m / time_seconds
        estimated_speed = speed_m_per_s * 3.6

        vehicle_id = work_locations[i][2]
        overspeed_flag = estimated_speed > speed_limit
        speed.append([round(estimated_speed, 1), vehicle_id, overspeed_flag])

        # Draw speed and warning on frame
        text = f"ID:{vehicle_id} {round(estimated_speed,1)}km/h"
        color = (0, 0, 255) if overspeed_flag else (0, 255, 0)  # Red if overspeed
        position = (int(x1), int(y1 - 10))  # Above the vehicle center
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if overspeed_flag:
            warning_text = "OVERSPEED!"
            cv2.putText(frame, warning_text, (int(x1), int(y1 - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return speed, frame


def draw_speed(img, speed, bbox_xywh, identities):
    '''
    Loop through each estimated speed (speed) and match it with the corresponding vehicle identity (identities), 
    then use its bounding box (bbox_xywh) to overlay the speed on the image.
    '''
    for i, j in enumerate(speed):
        for m, n in enumerate(identities):
            if j[1] == n: # Speed record's vehicle ID (j[1]) matches the current identity (n) in all tracked vehicles in the current frame
                xy = [int(i) for i in bbox_xywh[m]]
                cv2.putText(img, str(j[0]) + 'km/h', (xy[0], xy[1] - 7), cv2.FONT_HERSHEY_PLAIN, 1.5, [255, 255, 255],
                            2)
                break


def bbox_rel(image_width, image_height, *xyxy):
    '''
    Converts absolute corner coordinates to relative bounding box format:
    returns center x, center y, width, and height (all in pixels).
    '''
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    '''
    Simple function that adds fixed color depending on the class
    '''
    # Creating a tuple of large prime-like numbers that are used as multipliers in the color generation formula
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, cls_names, classes2, identities=None):
    '''
    Draws bounding boxes with labels and IDs on the image
    '''
    offset = (0, 0)
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(int(classes2[i] * 100))
        label = '%d %s' % (id, cls_names[i])
        # label +='%'
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

    return img


def drawing_violation_area(video_path):
    # -------------------------Extract background, display the first frame image, and mark the restricted lane -----------------------#
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    imag = cv2.imwrite('background.png', image)
    if imag:
        print('Background image extracted')
    # -------------------------------------Dedicated lane line drawing-----------------------------------#
    # 1.Read background image
    img = cv2.imread('background.png')
    num_points_to_mark = 4
    sl = []
    im = array(Image.open('background.png'))
    # 2.Draw points connected by solid lines and the corner points of a rectangle
    # Turn on interactive mode
    ion()
    imshow(im)
    title('Mark the Restricted Lane: Click 4 points in clockwise order, \nstarting from the top-left corner.')

    remaining_text = text(350, 900, f"You have {num_points_to_mark} points to mark.",
                          fontsize=12)

    for cs in range(num_points_to_mark):
        print(f'Click {num_points_to_mark - cs} more point(s)...')
        x = ginput(1)
        print('You clicked:', x)
        sl.append(x)

        # Update on-plot message
        remaining = num_points_to_mark - (cs + 1)
        remaining_text.set_text(f"{'Done! Close the window to proceed.' if remaining == 0 else f'{remaining} point(s) left to mark...'}")
        show()

    ioff()
    show()
    print(im.shape)

    # 3.Draw points connected by solid lines and the corner points of a rectangle
    no_entry_area = []

    # Compute the equation of the line
    def solid_line(x1, y1, x2, y2) -> int:
        if x1 == x2:
            k = -999
            b = 0
        else:
            k = (y2 - y1) / (x2 - x1)
            b = y1 - x1 * k
        return k, b

    m = 0
    n = 1
    for i in range(len(sl)):
        if n == 4:
            n = 0
        x = {}
        x1 = int(((sl[m])[0])[0])
        y1 = int(((sl[m])[0])[1])
        x2 = int(((sl[n])[0])[0])
        y2 = int(((sl[n])[0])[1])
        k, b = solid_line(x1, y1, x2, y2)
        if y1 > y2:
            yy = y2
            xx = x2
            y2 = y1
            x2 = x1
            y1 = yy
            x1 = xx
        if k != 0:
            for xxx in range(y1, y2):
                xq = (xxx - b) / k
                xq = int(xq)
                cv2.rectangle(img, (int(xq + 2), int(xxx)), (int(xq - 2), int(xxx)), (0, 0, 255), 2)
        else:
            for xxx in range(x1, x2):
                yq = b
                cv2.rectangle(img, (int(xxx), int(yq + 2)), (int(xxx), int(yq - 2)), (0, 0, 255), 2)
        x['k'] = k
        x['b'] = b
        x['x1'] = x1
        x['x2'] = x2
        x['y1'] = y1
        x['y2'] = y2
        print('k:', k, 'b:', b)
        no_entry_area.append(x)
        n += 1
        m += 1
    print(no_entry_area)
    cv2.imwrite('out.png', img)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Restricted area drawing")
    plt.axis("off")
    plt.show()
    # Write the obtained lane line information into car_line.txt
    with open("car_line.txt", 'w') as f:
        for s in no_entry_area:
            f.write(str(s) + '\n')

def detect(opt, save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size

    webcam = source == '0' or source.startswith(('rtsp', 'http')) or source.endswith('.txt')

    capture = cv2.VideoCapture(source)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_fature = (frame_width, frame_height)
    fps = capture.get(cv2.CAP_PROP_FPS)

    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE,
                        n_init=cfg.DEEPSORT.N_INIT,
                        nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=torch.cuda.is_available())

    device = select_device(opt.device)
    half = device.type != 'cpu'

    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, force_reload=False)
    model.to(device).eval()
    if half:
        model.half()

    accident_model = torch.load(opt.accident_weights, map_location=device)
    if isinstance(accident_model, dict):
        from models.experimental import attempt_load
        accident_model = attempt_load(opt.accident_weights, device=device)
    accident_model.eval()

     # Dataset
    if webcam:
        view_img = True
        cudnn.benchmark = True # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        view_img = True
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    names = model.names
    vehicle_names = {i: names[i] for i in VEHICLE_CLASSES if i < len(names)}

    save_path = str(Path(out))
    Path(save_path).mkdir(parents=True, exist_ok=True)
    txt_path = save_path + '/results.txt'
    vid_path, vid_writer = None, None

    up_counter = [0] * (max(VEHICLE_CLASSES.keys()) + 1)
    down_counter = [0] * (max(VEHICLE_CLASSES.keys()) + 1)
    counter_recording = []
    line_pixel = [frame_height // 2]
    width = {1: 0.64, 2: 1.85, 3: 0.9, 5: 2.3, 7: 2.5} #1: 'bicycle', 2: 'car',3: 'motorcycle',   5: 'bus',  7: 'truck'
    locations = []
    speed = []
    t0 = time.time()
    prev_tracks = {}

    for frame_idx, (path, img, im0s, vid_cap, _) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time_sync()
        results = model(img)
        pred = non_max_suppression(results, opt.conf_thres, opt.iou_thres,
                                   classes=list(VEHICLE_CLASSES.keys()), agnostic=opt.agnostic_nms)
        t2 = time_sync()

        for i, det in enumerate(pred):
            if webcam:
                p, s, im0 = path[i], f'{i}: ', im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            if det is None or not len(det):
                continue

            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()

            bbox_xywh, confs, classes = [], [], []
            img_h, img_w = im0.shape[:2]
            for *xyxy, conf, cls in det:
                x_c, y_c, w, h = bbox_rel(img_w, img_h, *xyxy)
                bbox_xywh.append([x_c, y_c, w, h])
                confs.append([conf.item()])
                classes.append([cls.item()])

            xywhs = torch.Tensor(bbox_xywh)
            confss = torch.Tensor(confs)
            classes_tensor = torch.Tensor(classes)

            track_outputs = deepsort.update(xywhs, confss, im0, classes_tensor)

            # Convert DeepSORT outputs to the format expected by find_accidents
            formatted_outputs = []
            for j, track in enumerate(track_outputs):
                x1, y1, x2, y2, track_id, class_id = *track[:4], int(track[4]), int(track[5])
                formatted_outputs.append([x1, y1, x2, y2, track_id, class_id])

            # Level 1: Detect accident
            is_crash, crash_index, prev_tracks = find_accidents(formatted_outputs, prev_tracks, frame_count_threshold=5)

            if is_crash:
                for (class1, class2) in is_crash:
                    cv2.putText(im0, f"POSSIBLE: {vehicle_names[class1]} - {vehicle_names[class2]} COLLISION",
                                (700, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)  # Yellow color

                # Level 2: Image Classification Verification
                img_pil = Image.fromarray(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                img_t = transform(img_pil).unsqueeze(0).to(device)

                with torch.no_grad():
                    clf_output = accident_model(img_t)
                    probabilities = torch.nn.functional.softmax(clf_output, dim=1)
                    conf, pred = torch.max(probabilities, 1)

                class_name = "Accident" if pred.item() == 0 else "Non-Accident"
                accident_confidence = probabilities[0][0].item()

                result_text = f'CLASSIFIER: {class_name} ({accident_confidence:.2f})'
                color = (0, 0, 255) if class_name == "Accident" else (0, 255, 0)
                for idx, (c1, c2) in enumerate(is_crash):
                    cv2.putText(im0, result_text, (700, 60 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                filename = str(Path(save_path) / f"collision_{time.strftime('%Y%m%d_%H%M%S')}_{class_name.lower()}_{accident_confidence:.2f}.jpg")
                cv2.imwrite(filename, im0)

            counter_recording, up_counter, down_counter, location, im0 = count_vehicles_and_check_restrictions(
                im0, track_outputs, line_pixel, counter_recording, up_counter, down_counter, VEHICLE_CLASSES
            )

            locations.append(location)
            if len(locations) == 5:
                if locations[0] and locations[-1]:
                    speed_data, frame = estimate_speed([locations[0], locations[-1]], fps, width, im0)
                    with open('speed.txt', 'a+') as speed_record:
                        for sp in speed_data:
                            speed_record.write(f'id:{sp[1]} {sp[0]}km/h\n')
                locations = []

            if len(track_outputs) > 0:
                bbox_xyxy = track_outputs[:, :4]
                identities = track_outputs[:, -2]
                classes2 = track_outputs[:, -1]
                draw_speed(im0, speed, bbox_xyxy, identities)
                draw_boxes(im0, bbox_xyxy, [vehicle_names[i] for i in classes2], classes2, identities)
                draw_up_down_counter(im0, up_counter, down_counter, frame_fature, vehicle_names)
                cv2.line(im0, (0, frame_fature[1] // 2), (frame_fature[0], frame_fature[1] // 2), (0, 0, 100), 2)

            if save_txt and len(track_outputs) != 0:
                for output in track_outputs:
                    with open(txt_path, 'a') as f:
                        f.write(('%g ' * 10 + '\n') % (frame_idx, output[-2], *output[:4], -1, -1, -1, -1))

            print('%sDone. (%.3fs)' % (s, t2 - t1))
            cv2.imshow(p, im0)
            if cv2.waitKey(1) == ord('q'):
                raise StopIteration

            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(str(Path(save_path) / Path(p).name), im0)
                else:
                    if vid_path != p:
                        vid_path = p
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        fourcc = cv2.VideoWriter_fourcc(*opt.fourcc)
                        vid_writer = cv2.VideoWriter(str(Path(save_path) / Path(p).name), fourcc, fps, (frame_width, frame_height))
                    vid_writer.write(im0)

            with open('counter.txt', 'w') as counter:
                counter.write('up:%s\ndown:%s' % (str(up_counter), str(down_counter)))

            if save_txt or save_img:
                print('Results saved to %s' % (os.getcwd() + os.sep + out))

            print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='model.pt path')  # vehicle detection model
    parser.add_argument('--accident-weights', type=str, default='fineTunedClassifier.pt', help='accident classification model path') 
    parser.add_argument('--source', type=str, default='TC5/video2.mp4', help='source')  # input video
    parser.add_argument('--output', type=str, default='TC5_output/video2_output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, default=[0, 1, 2], help='filter by class')  # car、truck、bus
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference') # 
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    args = parser.parse_args()
    # Verify whether the input image size is a multiple of 32
    args.img_size = check_img_size(args.img_size)
    print(args)

    # Tensors under this statement will not track gradients, which saves memory
    with torch.no_grad():  
        # Function to draw restricted area
        drawing_violation_area(args.source)

        # Main function for speed estimation, vehicle counting, collision detection, and check whether vehicles entered violation area
        detect(args)