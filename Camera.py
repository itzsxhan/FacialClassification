import cv2

def CamShot():
    global img_name
    camera = cv2.VideoCapture(0)

    cv2.namedWindow("Web-Cam")

    img_counter = 0

    while True:
        ret, frame = camera.read()

        if not ret:
            print("failed to grab image")
            break
        cv2.imshow("test",frame)

        keyBind = cv2.waitKey(1)

        if keyBind%256 == 27:
            print("webcam closing")
            break
        if keyBind%256 == 32:
            img_name = "image_ai_{}.png".format(img_counter)
            cv2.imwrite(img_name,frame)
            print("screenshot-taken")
            img_counter += 1

        if img_counter == 1:
            break

    camera.release()
    cv2.destroyWindow("Web-Cam")
    return img_name

