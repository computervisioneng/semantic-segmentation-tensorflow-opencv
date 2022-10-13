def get_detections(net, blob):
    net.setInput(blob)
    boxes, masks = net.forward(["detection_out_final", "detection_masks"])
    return boxes, masks
