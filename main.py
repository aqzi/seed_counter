import cv2
import supervision as sv
import inference

model = inference.get_model("seeds-ptw6s/2")

def count_seeds():
    image = cv2.imread("data/images/0002.jpg")

    results = model.infer(image)[0]

    detections = sv.Detections.from_inference(results)

    mask_annotator = sv.BoxAnnotator(color=sv.Color.YELLOW, thickness=2)

    annotated_image = mask_annotator.annotate(
        scene=image, detections=detections)
    
    amount_of_seeds = len(results.predictions)

    # Calculate text size and position
    text = f"Number of seeds: {amount_of_seeds}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.5
    thickness = 3
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = text_size[1] + 20  # 20 pixels padding from the top

    # Draw the text
    cv2.putText(annotated_image, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)

    # Display the image
    cv2.imshow("Detections", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #save the image
    cv2.imwrite("data/annotated/0000.jpg", annotated_image)


count_seeds()