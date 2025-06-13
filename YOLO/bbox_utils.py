def xywh_to_xyxy(bbox):
    """
    Converting bbox from xywh format to xyxy format(easier to calculate iou using this)
    Saw smth like this in the source code of DEEPSORT just some days back
    """
    bbox_x,bbox_y,bbox_w,bbox_h,bbox_confidence=bbox

    x_max=bbox_x+(bbox_w/2)
    x_min=bbox_x-(bbox_w/2)
    y_max=bbox_y+(bbox_h/2)
    y_min=bbox_y-(bbox_h/2)

    return x_min,y_min,x_max,y_max


def iou(bbox_predicted,bbox_actual):

    """
    One thing to note here about the coordinates which even I got a bit confused a bout at the beginning is top left means (0,0)
    So as we move towards the right the x value increases(which is actually pretty normal) but the thing which i didnt realize initially
    is that as we move downwards value of y increases(which is the opposite of how we normally percieve y values)

    """
    x_min_predicted,y_min_predicted,x_max_predicted,y_max_predicted=xywh_to_xyxy(bbox_predicted)
    x_min_actual,y_min_actual,x_max_actual,y_max_actual=xywh_to_xyxy(bbox_actual)

    x_intersection_right=min(x_max_actual,x_max_predicted)
    x_intersection_left=max(x_min_actual,x_min_predicted)
    y_intersection_lower=min(y_max_actual,y_max_predicted)
    y_intersection_upper=max(y_min_actual,y_min_predicted)

    w_inter=max(0,x_intersection_right-x_intersection_left)
    h_inter=max(0,y_intersection_lower-y_intersection_upper)

    w_box1=max(0,x_max_predicted-x_min_predicted)
    h_box1=max(0,y_max_predicted-y_min_predicted)

    w_box2=max(0,x_max_actual-x_min_actual)
    h_box2=max(0,y_max_actual-y_min_actual)

    area_box1=w_box1*h_box1

    area_box2=w_box2*h_box2
    
    area_inter=w_inter*h_inter

    area_union=area_box1+area_box2-area_inter

    if area_union==0:
        return 0
    else:
        iou=area_inter/area_union
        return iou

#Test

box1 = [50, 50, 40, 40, 1.0]  
box2 = [50, 50, 20, 20, 1.0] 


# print("IoU:", iou(box1, box2)) 