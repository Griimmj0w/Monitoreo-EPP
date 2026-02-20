import cv2


class HelmetTagReader:
    def __init__(self):
        self.qr = cv2.QRCodeDetector()

    def read_qr_from_crop(self, crop):
        if crop is None or getattr(crop, "size", 0) == 0:
            return None

        decoded, pts, _ = self.qr.detectAndDecode(crop)
        if decoded:
            return decoded.strip()
        return None

import cv2

class HelmetTagReader:
    def __init__(self):
        self.qr = cv2.QRCodeDetector()

    def read_qr_from_crop(self, crop):
        if crop is None or crop.size == 0:
            return None
        
        decoded, pts, _ = self.qr.detectAndDecode(crop)
        if decoded:
            return decoded.strip()
        return None
    
