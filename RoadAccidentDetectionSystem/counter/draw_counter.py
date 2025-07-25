import cv2

def draw_up_down_counter(img, up_counter, down_counter, frame_feature, names):
    # Draw background box
    cv2.rectangle(img, (0, 0), (260, 30 + 20 * (len(names) + 2)), (255, 255, 255), thickness=-1)

    # Header
    cv2.putText(img, 'veh_type', (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
    text_size = cv2.getTextSize('veh_type', cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, thickness=-1)
    offset_up = int(text_size[0][0]) + 30
    offset_down = offset_up + 60

    cv2.putText(img, 'up', (offset_up, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
    cv2.putText(img, 'down', (offset_down, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)

    # Vehicle-specific counters
    for i, class_id in enumerate(names):
        name = names[class_id]
        y_pos = (i + 2) * 20
        up = up_counter[class_id] if class_id < len(up_counter) else 0
        down = down_counter[class_id] if class_id < len(down_counter) else 0

        cv2.putText(img, name, (10, y_pos), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
        cv2.putText(img, str(up), (offset_up, y_pos), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
        cv2.putText(img, str(down), (offset_down, y_pos), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)

    # Total count
    total_up = sum(up_counter[:len(names)])
    total_down = sum(down_counter[:len(names)])
    y_pos = (len(names) + 2) * 20

    cv2.putText(img, 'Total', (10, y_pos), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
    cv2.putText(img, str(total_up), (offset_up, y_pos), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
    cv2.putText(img, str(total_down), (offset_down, y_pos), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
