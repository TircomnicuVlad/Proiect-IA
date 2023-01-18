import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract'

#initialize the video capture
cap = cv2.VideoCapture('Model/Test.mp4')

while True:
    # read the next frame
    ret, frame = cap.read()
    if not ret:
        break

    # pre-processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        aspectRatio = w / float(h)
        if (aspectRatio > 2.5 and aspectRatio < 6.0):
            # license_plate found
            license_plate = frame[y: y + h, x: x+w]
            # extract the text using OCR
            text = pytesseract.image_to_string(license_plate, lang='eng', config='--psm 11')
            print(text)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()