import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
import time
import serial
import serial.tools.list_ports

class VideoCaptureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Capture App")
        
        self.num_videos = tk.IntVar(value=1)
        self.duration = tk.IntVar(value=10)
        self.elapsed_time = tk.IntVar(value=5)
        self.save_folder = tk.StringVar()
        self.settings_file = tk.StringVar()
        self.frame_rate = tk.DoubleVar()
        self.image_width = tk.IntVar()
        self.image_height = tk.IntVar()
        self.usb_port = tk.StringVar()
        self.selected_laser = tk.StringVar(value="LASER 1")
        
        self.create_widgets()
        
    def create_widgets(self):
        tk.Label(self.root, text="Number of Videos:").grid(row=0, column=0)
        tk.Entry(self.root, textvariable=self.num_videos).grid(row=0, column=1)
        
        tk.Label(self.root, text="Duration (seconds):").grid(row=1, column=0)
        tk.Entry(self.root, textvariable=self.duration).grid(row=1, column=1)
        
        tk.Label(self.root, text="Elapsed Time Between Videos (seconds):").grid(row=2, column=0)
        tk.Entry(self.root, textvariable=self.elapsed_time).grid(row=2, column=1)
        
        tk.Button(self.root, text="Browse Save Folder", command=self.browse_save_folder).grid(row=3, column=0, columnspan=2)
        tk.Label(self.root, textvariable=self.save_folder).grid(row=4, column=0, columnspan=2)
        
        tk.Button(self.root, text="Load Settings from File", command=self.load_settings_from_file).grid(row=5, column=0, columnspan=2)
        tk.Label(self.root, textvariable=self.settings_file).grid(row=6, column=0, columnspan=2)
        
        tk.Label(self.root, text="Frame Rate (fps):").grid(row=7, column=0)
        tk.Entry(self.root, textvariable=self.frame_rate).grid(row=7, column=1)
        
        tk.Label(self.root, text="Image Width:").grid(row=8, column=0)
        tk.Entry(self.root, textvariable=self.image_width).grid(row=8, column=1)
        
        tk.Label(self.root, text="Image Height:").grid(row=9, column=0)
        tk.Entry(self.root, textvariable=self.image_height).grid(row=9, column=1)
        
        tk.Label(self.root, text="USB Port:").grid(row=10, column=0)
        tk.Entry(self.root, textvariable=self.usb_port).grid(row=10, column=1)
        
        tk.Button(self.root, text="Detect Arduino Port", command=self.detect_arduino_port).grid(row=11, column=0, columnspan=2)
        
        tk.Label(self.root, text="Select Laser:").grid(row=12, column=0)
        lasers = ["LASER 1", "LASER 2", "LASER 3", "LASER 4"]
        for i, laser in enumerate(lasers, start=13):
            tk.Radiobutton(self.root, text=laser, variable=self.selected_laser, value=laser).grid(row=i, column=0, columnspan=2)
        
        tk.Button(self.root, text="Start Capture", command=self.start_capture).grid(row=17, column=0, columnspan=2)
        
        tk.Button(self.root, text="Preview Video", command=self.preview_video).grid(row=18, column=0, columnspan=2)
        
        tk.Button(self.root, text="Exit", command=self.root.quit).grid(row=19, column=0, columnspan=2)
        
    def browse_save_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.save_folder.set(folder_selected)
    
    def detect_arduino_port(self):
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if "Arduino" or "CH340" in port.description:
                self.usb_port.set(port.device)
                messagebox.showinfo("Info", f"Arduino detected on port {port.device}")
                return
        messagebox.showwarning("Warning", "Arduino not detected. Please enter the USB port manually.")
    
    def load_settings_from_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if not file_path:
            return
        
        self.settings_file.set(file_path)
        
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    if "Number of Videos" in line:
                        self.num_videos.set(int(line.split(":")[1].strip()))
                    elif "Duration (seconds)" in line:
                        self.duration.set(int(line.split(":")[1].strip()))
                    elif "Elapsed Time Between Videos (seconds)" in line:
                        self.elapsed_time.set(int(line.split(":")[1].strip()))
                    elif "Frame Rate (fps)" in line:
                        self.frame_rate.set(float(line.split(":")[1].strip()))
                    elif "Image Width" in line:
                        self.image_width.set(int(line.split(":")[1].strip()))
                    elif "Image Height" in line:
                        self.image_height.set(int(line.split(":")[1].strip()))
        except Exception as e:
            messagebox.showerror("Error", f"Could not read settings file: {e}")
            return
        
        messagebox.showinfo("Info", "Settings loaded successfully.")
        
    def start_capture(self):
        if not self.save_folder.get():
            messagebox.showerror("Error", "Please select a folder to save the videos.")
            return
        
        if not self.usb_port.get():
            messagebox.showerror("Error", "Please enter or detect the USB port for Arduino.")
            return
        
        try:
            arduino = serial.Serial(self.usb_port.get(), 9600)
        except serial.SerialException as e:
            messagebox.showerror("Error", f"Could not open serial port: {e}")
            return
        
        num_videos = self.num_videos.get()
        duration = self.duration.get()
        elapsed_time = self.elapsed_time.get()
        save_folder = self.save_folder.get()
        
        info_filename = os.path.join(save_folder, "video_info.txt")
        with open(info_filename, 'w') as info_file:
            info_file.write(f"Number of Videos: {num_videos}\n")
            info_file.write(f"Duration (seconds): {duration}\n")
            info_file.write(f"Elapsed Time Between Videos (seconds): {elapsed_time}\n")
            info_file.write(f"Frame Rate (fps): {self.frame_rate.get()}\n")
            info_file.write(f"Image Width: {self.image_width.get()}\n")
            info_file.write(f"Image Height: {self.image_height.get()}\n")
        
        for i in range(num_videos):
            video_folder = os.path.join(save_folder, f"video_{i+1}")
            os.makedirs(video_folder, exist_ok=True)
            
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                messagebox.showerror("Error", "Could not open video device.")
                arduino.close()
                return
            
            cap.set(cv2.CAP_PROP_FPS, self.frame_rate.get())
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width.get())
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height.get())
            
            selected_laser = self.selected_laser.get()
            
            # Display the selected laser message 2 seconds before capturing
            print(f"{selected_laser} on")
            arduino.write(f'{selected_laser} on\n'.encode())
            time.sleep(2)
            
            total_frames = int(self.frame_rate.get() * duration)
            
            t1 = time.time()

            for j in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_filename = os.path.join(video_folder, f"frame_{j+1:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
            
            t2 = time.time()
            print('fps: %.2f' % ( 1 / ( (t2 - t1) / total_frames) ) )
              
            # Display the selected laser message at the end of capturing
            print(f"{selected_laser} off")
            arduino.write(f'{selected_laser} off\n'.encode())
            
            cap.release()
            cv2.destroyAllWindows()
            
            if i < num_videos - 1:
                time.sleep(elapsed_time)  # Wait for elapsed_time seconds before the next video
        
        arduino.close()
        messagebox.showinfo("Info", "Video capture completed.")
        
    def preview_video(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open video device.")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow('Preview', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    root = tk.Tk()
    app = VideoCaptureApp(root)
    root.mainloop()


