def iou(bbox_predicted,bbox_actual):

    """
    One thing to note here about the coordinates which even I got a bit confused a bout at the beginning is top left means (0,0)
    So as we move towards the right the x value increases(which is actually pretty normal) but the thing which i didnt realize initially
    is that as we move downwards value of y increases(which is the opposite of how we normally percieve y values)

    """
    x_min_predicted,y_min_predicted,x_max_predicted,y_max_predicted=bbox_coordinate_format(bbox_predicted)
    x_min_actual,y_min_actual,x_max_actual,y_max_actual=bbox_coordinate_format(bbox_actual)

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

def bbox_coordinate_format(bbox):
    """
    So basically the bbox coordinate predicted by the model are bw 0 and 1

    eg x = 0.5 means  box is centered horizontally at the middle of the image 
    y = 0.5 means box is centered vertically at the middle
    w = 0.2 means the box takes up 20% of the image width
    h = 0.3 means the box takes up 30% of image height

    This fn maps these values to the actual bbox coordinates in the image
    """
    
    img_width, img_height = 448, 448

    x,y,w,h,_=bbox

    x_center = x * img_width
    y_center = y * img_height
    box_width = w * img_width
    box_height = h * img_height

    x1 = x_center - box_width / 2
    y1 = y_center - box_height / 2
    x2 = x_center + box_width / 2
    y2 = y_center + box_height / 2

    return x1.item(),y1.item(),x2.item(),y2.item()

#Test

box1 = [0.5, 0.5, 0.3, 0.3, 1.0]  # Centered box taking 30% width/height
box2 = [0.5, 0.5, 0.2, 0.2, 1.0]  # Smaller box in center



