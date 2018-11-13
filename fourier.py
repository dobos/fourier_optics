import cv2
import gc
import os
import numpy as np

class Filter:
    HIGHPASS = 1
    LOWPASS = 0
    
    def __init__(self, t=LOWPASS):
        self.type = t
        self.size1 = 16
        self.size2 = 32
        self.changed = True
        self.value = None
        self.gui = []
                
    def set_type(self, t):
        self.type = t
        self.changed = True
        
    def set_size1(self, size1):
        self.size1 = size1
        self.changed = True
        
    def set_size2(self, size2):
        self.size2 = size2
        self.changed = True
        
    def create(self):
        filt = np.full((SIZE, SIZE), 1)
        return filt
            
    def get(self):
        if self.changed:
            self.value = self.create()
            self.changed = False
        return self.value
    

class RectFilter(Filter):
    def __init__(self, t=Filter.LOWPASS, width=16, height=16):
        super(self.__class__, self).__init__(t)
        self.size1 = width
        self.size2 = height
    
    def create(self):
        mid = int(SIZE / 2)
        filt = np.full((SIZE, SIZE), self.type)
        filt[mid - self.size2 : mid + self.size2, mid - self.size1 : mid + self.size1] = (self.type + 1) % 2
        return filt
    
class CircleFilter(Filter):
    def __init__(self, t=Filter.LOWPASS, inner=16, outer=32):
        super(self.__class__, self).__init__(t)
        self.size1 = inner
        self.size2 = outer
    
    def circle_mask(xx, yy, r):
        mask = np.full((SIZE, SIZE), False)
        for x in range(0, SIZE):
            for y in range(0, SIZE):
                mask[x, y] = (x - xx)**2 + (y - yy)**2 < r**2
        return mask
    
    def create(self):
        mid = int(SIZE / 2)
        mask1 = CircleFilter.circle_mask(mid, mid, self.size1)
        mask2 = CircleFilter.circle_mask(mid, mid, self.size2)
        filt = np.full((SIZE, SIZE), self.type)
        filt[mask2] = (self.type + 1) % 2
        filt[mask1] = self.type
        return filt
    
class CheckFilter(Filter):
    def __init__(self, t=Filter.LOWPASS, size=16):
        super(self.__class__, self).__init__(t)
        self.size1 = size
            
    def create(self):
        filt = np.zeros((SIZE,SIZE))
        step1 = max(1, int(self.size1 // 2))
        step2 = max(1, int(self.size2 // 2))
        for x in range(0, int(SIZE / step1)):
            for y in range(0, int(SIZE / step2)):
               filt[x * step1 : (x + 1) * step1, y * step2 : (y + 1) * step2] = (x + y + self.type) % 2
        filt = np.roll(filt, int(step1 / 2), axis=0)
        filt = np.roll(filt, int(step2 / 2), axis=1)
        return filt
		
class Fourier():
    def __init__(self):
        self.filter = None
        self.set_filter(1)
    
    def capture_single_frame(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        frame = format_frame(frame)
        cap.release()
        return frame

    def capture_frame(self, cap):
        # Capture frame, convert to B&W and crop to 256 by 256
        ret, frame = cap.read()
        if ret:
            frame = self.format_frame(frame)
            return frame
        else:
            return None

    def format_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame[CROP[0][0]:CROP[0][1], CROP[1][0]:CROP[1][1]]
        frame = frame / IMAX
        return frame

    def fft_frame(self, frame):
        spec = np.fft.fft2(frame)
        spec = np.fft.fftshift(spec)
        return spec

    def filter_frame(self, spec, filt):
        spec = np.fft.ifftshift(spec * filt)
        tran = np.abs(np.fft.ifft2(spec))
        return tran

    def scale_frame(self, frame):
        fmax = np.max(frame)
        fmin = np.min(frame)
        return np.uint8(IMAX * (frame - fmin) / (fmax - fmin))

    def combine_image(self, frame, spec, filt, tran):
        disp = np.empty((2 * SIZE, 2 * SIZE), dtype=np.uint8)
        disp[0:SIZE,0:SIZE] = self.scale_frame(frame)
        disp[SIZE:,SIZE:] = self.scale_frame(tran)
        s = self.scale_frame(np.log(np.abs(spec)))
        disp[SIZE:,0:SIZE] = s
        disp[0:SIZE,SIZE:] = s * filt
        return disp
    
    def create_gui(self):
        wnd = 'image'
        cv2.namedWindow(wnd, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(wnd, 2 * SIZE, 2 * SIZE)

        switch = 'FILTER: 0 = DISABLED | 1 = RECTANGLE | 2 = CIRCLE/RING'
        cv2.createTrackbar(switch, wnd, 1, 2, self.set_filter)
        switch = 'TYPE: 0 = LOW PASS | 1 = HIGH PASS'
        cv2.createTrackbar(switch, wnd, self.filter.type, 1, self.set_type)
        cv2.createTrackbar('SIZE 1', wnd, self.filter.size1, int(SIZE / 2), self.set_size1)
        cv2.createTrackbar('SIZE 2', wnd, self.filter.size2, int(SIZE / 2), self.set_size2)
        
        return wnd
        
    def destroy_gui(self):
        cv2.destroyAllWindows()
        
    def set_filter(self, f):
        if f == 0:
            ff = Filter()
        elif f == 1:
            ff = RectFilter()
        elif f == 2:
            ff = CircleFilter()
        elif f == 3:
            ff = CheckFilter()

        if self.filter:
            ff.type = self.filter.type
            ff.size1 = self.filter.size1
            ff.size2 = self.filter.size2

        self.filter = ff
            
    def set_type(self, t):
        self.filter.set_type(t)
        
    def set_size1(self, size1):
        self.filter.set_size1(size1)
        
    def set_size2(self, size2):
        self.filter.set_size2(size2)
    
    def play_video(self):
        cap = cv2.VideoCapture(CAMERA)
        wnd = self.create_gui()

        while(True):
            frame = self.capture_frame(cap)
            spec = self.fft_frame(frame)
            filt = self.filter.get()
            tran = self.filter_frame(spec, filt)

            # Combine parts into a single frame
            disp = self.combine_image(frame, spec, filt, tran)

            # Display frame
            cv2.imshow('image', disp)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        self.destroy_gui()
        cap.release()
        gc.collect() 

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
CAMERA = 0
IMAX = 255

SIZE = 256
CROP = [[112, 368], [192, 448]]

#SIZE = 480
#CROP = [[0, 480], [0, 480]]


fourier = Fourier()
fourier.play_video()