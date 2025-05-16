import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageOps
import cv2
import threading
import pygame
from playsound import playsound
from main2 import yolo_model, cnn_model, food_labels, detect_and_crop_food, predict_food

# ==== Giá món ăn ====
prices = {
    'ca_hu_kho': 35000, 'canh_bi_do': 40000, 'canh_bi_xanh': 30000, 'canh_cai': 20000,
    'canh_chua': 30000, 'canh_khoai_mon': 25000, 'cha': 35000, 'com': 100,
    'dau_hu_sot_ca': 45000, 'dua_leo': 40, 'ga_chien': 50000, 'ga_kho': 25000,
    'lap_xuong': 25000, 'mam_nem': 30, 'nuoc_mam': 10, 'rau': 5000,
    'rau_muong_xao': 20000, 'thit_kho': 25000, 'thit_kho_trung': 25000,
    'tom_kho': 25000, 'trung-chien': 25000, 'xi_dau': 50
}

# ==== Phát âm thanh click ====
def play_click():
    threading.Thread(target=lambda: playsound("pict/click.mp3"), daemon=True).start()

# ==== Phát nhạc nền ====
def play_background_music():
    try:
        pygame.mixer.init()
        pygame.mixer.music.load("pict/nhạc nền.mp3")
        pygame.mixer.music.set_volume(0.3)
        pygame.mixer.music.play(-1)  # Lặp vô hạn
    except Exception as e:
        print("Lỗi phát nhạc nền:", e)

def smart_background(path, screen_w, screen_h, bg_color="white"):
    img = Image.open(path)
    if img.width >= screen_w or img.height >= screen_h:
        return ImageOps.fit(img, (screen_w, screen_h), method=Image.Resampling.LANCZOS)
    else:
        return ImageOps.pad(img, (screen_w, screen_h),
                            method=Image.Resampling.LANCZOS,
                            color=bg_color, centering=(0.5, 0.5))

class WelcomeScreen(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        bg_img = smart_background("pict/d.png", screen_w, screen_h)

        self.bg_photo = ImageTk.PhotoImage(bg_img)
        tk.Label(self, image=self.bg_photo).place(x=0, y=0, relwidth=1, relheight=1)

        tk.Button(self, text="PRESS HERE", font=("Arial", 20, "bold"),
                  bg="green", fg="white", width=12,
                  command=lambda: [play_click(), controller.show_frame("FoodGUI")])\
            .place(relx=0.42, rely=0.76)

class FoodGUI(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.cart = {}
        self.freeze = False
        self.last_frame = None
        self.cap = None
        self.fullscreen = False
        self.music_on = True
        self.selected_cam = tk.IntVar(value=0)

        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        bg_img = smart_background("pict/bg.png", screen_w, screen_h)
        self.bg_photo = ImageTk.PhotoImage(bg_img)
        tk.Label(self, image=self.bg_photo).place(x=0, y=0, relwidth=1, relheight=1)

        self.cam_menu = ttk.Combobox(self, values=self.detect_cameras(),
                                     textvariable=self.selected_cam, width=5)
        self.cam_menu.place(x=10, y=10)
        self.cam_menu.bind("<<ComboboxSelected>>", lambda e: self.set_camera(self.selected_cam.get()))

        self.frame_label = tk.Label(self, bg="black")
        self.frame_label.place(relx=0.05, rely=0.35, relwidth=0.4, relheight=0.5)

        self.bill_box = tk.Text(self, font=("Arial", 12), bg="white", fg="black")
        self.bill_box.place(relx=0.5, rely=0.35, relwidth=0.4, relheight=0.5)

        self.total_label = tk.Label(self, text="TOTAL COST: 0 VND", bg="orange",
                                    fg="white", font=("Arial", 14, "bold"))
        self.total_label.place(relx=0.5, rely=0.82, relwidth=0.4, height=40)

        # ==== Các nút với âm thanh ====
        btn_y = 0.88
        tk.Button(self, text="SCAN", font=("Arial", 10, "bold"),
                  command=lambda: [play_click(), self.detect()])\
            .place(relx=0.5, rely=btn_y, width=80)

        tk.Button(self, text="NEXT", font=("Arial", 10, "bold"),
                  command=lambda: [play_click(), self.resume_camera()])\
            .place(relx=0.61, rely=btn_y, width=80)

        tk.Button(self, text="DELETE MEAL", font=("Arial", 10, "bold"),
                  command=lambda: [play_click(), self.clear_cart()])\
            .place(relx=0.72, rely=btn_y, width=110)

        tk.Button(self, text="HOME", font=("Arial", 10, "bold"), bg="red", fg="white",
                  command=lambda: [play_click(), self.go_home()])\
            .place(relx=0.9, rely=0.05, width=100)

        tk.Button(self, text="MUSIC", font=("Arial", 10, "bold"), bg="blue", fg="white",
                  command=lambda: [play_click(), self.toggle_music()])\
            .place(relx=0.78, rely=0.05, width=100)

        self.root_bindings()
        self.update_frame()

    def toggle_music(self):
        if self.music_on:
            pygame.mixer.music.pause()
        else:
            pygame.mixer.music.unpause()
        self.music_on = not self.music_on

    def on_show(self):
        self.cart.clear()
        self.freeze = False
        self.set_camera(self.selected_cam.get())
        self.update_bill()

    def root_bindings(self):
        self.bind_all("<F11>", self.toggle_fullscreen)
        self.bind_all("<Escape>", lambda e: self.controller.destroy())
        self.bind_all("<space>", lambda e: self.detect())

    def detect_cameras(self, max_devices=5):
        return [i for i in range(max_devices) if cv2.VideoCapture(i).isOpened()] or [0]

    def set_camera(self, cam_index):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(int(cam_index))

    def toggle_fullscreen(self, event=None):
        self.fullscreen = not self.controller.attributes("-fullscreen", self.fullscreen)

    def update_frame(self):
        if not self.freeze and self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.last_frame = frame.copy()
                frame = cv2.resize(frame, (640, 480))
                img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                self.frame_label.configure(image=img)
                self.frame_label.imgtk = img
        self.after(20, self.update_frame)

    def detect(self):
        if self.last_frame is None: return
        self.freeze = True
        frame = self.last_frame.copy()
        crops = detect_and_crop_food(frame)
        annotated = frame.copy()

        for crop, yolo_conf, yolo_label in crops:
            cnn_label, cnn_conf = predict_food(crop, cnn_model, food_labels)
            label = yolo_label if yolo_conf >= 0.6 else cnn_label if cnn_conf >= 0.75 else "Unknown"
            if label != "Unknown":
                self.cart[label] = self.cart.get(label, 0) + 1

        results = yolo_model(frame, conf=0.6)[0]
        for box, cls_id in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy().astype(int)):
            x1, y1, x2, y2 = map(int, box)
            label = food_labels[cls_id]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        show = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(cv2.resize(annotated, (640, 480)), cv2.COLOR_BGR2RGB)))
        self.frame_label.configure(image=show)
        self.frame_label.imgtk = show
        self.update_bill()

    def resume_camera(self):
        self.freeze = False

    def clear_cart(self):
        self.cart.clear()
        self.freeze = False
        self.update_bill()

    def update_bill(self):
        self.bill_box.delete("1.0", tk.END)
        total = 0
        for item, qty in self.cart.items():
            price = prices.get(item, 0)
            self.bill_box.insert(tk.END, f"{item} x{qty} = {price*qty:,} VND\n")
            total += price * qty
        self.total_label.config(text=f"TOTAL COST: {total:,} VND")

    def go_home(self):
        if self.cap:
            self.cap.release()
        self.controller.show_frame("WelcomeScreen")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Smart Canteen")
        self.geometry("1366x768")
        self.frames = {}

        container = tk.Frame(self)
        container.pack(fill="both", expand=True)

        for FrameClass in (WelcomeScreen, FoodGUI):
            name = FrameClass.__name__
            frame = FrameClass(container, self)
            self.frames[name] = frame
            frame.place(x=0, y=0, relwidth=1, relheight=1)

        self.show_frame("WelcomeScreen")

    def show_frame(self, name):
        frame = self.frames[name]
        frame.tkraise()
        if hasattr(frame, "on_show"):
            frame.on_show()

# ===== KHỞI CHẠY APP =====
if __name__ == "__main__":
    play_background_music()
    app = App()
    app.mainloop()
    pygame.mixer.music.stop()
    pygame.mixer.quit()
