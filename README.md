# Khai báo thư viện
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageOps
import cv2
import threading
import pygame
from playsound import playsound
from main2 import yolo_model, cnn_model, food_labels, detect_and_crop_food, predict_food

# Load mô hình nhận diện sản phẩm (được tổng hợp trong file main2.py, bao gồm các file .h5 và .pt (tự huấn luyện dựa trên file ảnh thu thập(dataset) để detect và crop sản phẩm))
from main2 import yolo_model, cnn_model, food_labels, detect_and_crop_food, predict_food

# Mô hình detect và drop + dataset 
# https://drive.google.com/file/d/1jpeuNHOTjRNuLJNSLxSF-jwale0tXIxo/view?usp=sharing    

# File ảnh giao diện và âm thanh 
# https://drive.google.com/drive/folders/1KUu3Y0TFoNPhGkcB6RBawbdy3NJ5pjiy?usp=sharing

# Danh sách nhãn và giá của các loại sản phẩm
prices = {
    'ca_hu_kho': 35000, 'canh_bi_do': 40000, 'canh_bi_xanh': 30000, 'canh_cai': 20000,
    'canh_chua': 30000, 'canh_khoai_mon': 25000, 'cha': 35000, 'com': 100,
    'dau_hu_sot_ca': 45000, 'dua_leo': 40, 'ga_chien': 50000, 'ga_kho': 25000,
    'lap_xuong': 25000, 'mam_nem': 30, 'nuoc_mam': 10, 'rau': 5000,
    'rau_muong_xao': 20000, 'thit_kho': 25000, 'thit_kho_trung': 25000,
    'tom_kho': 25000, 'trung-chien': 25000, 'xi_dau': 50
}

# Phát âm thanh và giao diện
def play_click():
    threading.Thread(target=lambda: playsound("pict/click.mp3"), daemon=True).start()

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


# Lớp chính điều khiển giao diện và các thao tác xử lý (class FoodGUI)
class FoodGUI:
# Hàm khởi tạo giao diện
    def __init__(self, root):
        self.root = root
        self.root.title("HỆ THỐNG TÍNH TIỀN CANTEEN")
        self.cart = {}
        self.freeze = False
        self.cap = None
        self.last_frame = None


        # === CHỌN WEBCAM ===
        self.selected_cam = tk.IntVar(value=0)
        cam_frame = tk.Frame(self.root)
        cam_frame.grid(row=0, column=0, columnspan=2, sticky="w", pady=5)


        tk.Label(cam_frame, text="Chọn webcam:").pack(side=tk.LEFT)
        self.cam_menu = tk.OptionMenu(cam_frame, self.selected_cam, *self.detect_cameras(), command=self.set_camera)
        self.cam_menu.pack(side=tk.LEFT)


        # === KHUNG CAMERA ===
        self.frame_label = tk.Label(self.root)
        self.frame_label.grid(row=1, column=0, rowspan=10)


        # === HÓA ĐƠN ===
        tk.Label(self.root, text="Hóa đơn", font=("Arial", 14)).grid(row=1, column=1)
        self.bill_box = tk.Text(self.root, width=40, height=20)
        self.bill_box.grid(row=2, column=1, rowspan=6)


        # === NÚT BẤM ===
        btn_frame = tk.Frame(self.root)
        btn_frame.grid(row=8, column=1, pady=5)


        tk.Button(btn_frame, text=" Quét món ăn", command=self.detect).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text=" Tiếp", command=self.resume_camera).pack(side=tk.LEFT, padx=5)
        tk.Button(self.root, text=" Xóa giỏ hàng", command=self.clear_cart).grid(row=9, column=1)


        self.total_label = tk.Label(self.root, text="Tổng tiền: 0 VND", font=("Arial", 12))
        self.total_label.grid(row=10, column=1, pady=10)


        self.set_camera(self.selected_cam.get())
        self.update_frame()

# Hàm phát hiện webcam
    def detect_cameras(self, max_devices=5):
        available = []
        for i in range(max_devices):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available if available else [0]

# Hàm thiết lập camera
    def set_camera(self, cam_index):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(int(cam_index))

# Hàm cập nhật hình ảnh liên tục
    def update_frame(self):
        if not self.freeze and self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.last_frame = frame.copy()
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                imgtk = ImageTk.PhotoImage(image=img_pil)
                self.frame_label.imgtk = imgtk
                self.frame_label.configure(image=imgtk)
        self.root.after(20, self.update_frame)

# Hàm nhận diện món ăn
    def detect(self):
        self.freeze = True
        frame = self.last_frame.copy()
        crops = detect_and_crop_food(frame)
        annotated_frame = frame.copy()


        for crop, yolo_conf, yolo_label in crops:
            cnn_label, cnn_conf = predict_food(crop, cnn_model, food_labels)
            label = yolo_label if yolo_conf >= 0.6 else cnn_label if cnn_conf >= 0.75 else "Unknown"
            if label != "Unknown":
                self.cart[label] = self.cart.get(label, 0) + 1


        results = yolo_model(frame, conf=0.6)[0]
        for box, cls_id in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy().astype(int)):
            x1, y1, x2, y2 = map(int, box)
            label = food_labels[cls_id]
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(annotated_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)


        img_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=img_pil)
        self.frame_label.imgtk = imgtk
        self.frame_label.configure(image=imgtk)


        self.update_bill()

# Hàm tiếp tục camera
    def resume_camera(self):
        self.freeze = False

# Hàm cập nhật hóa đơn(hiện thị món, tính tổng tiền)
    def update_bill(self):
        self.bill_box.delete("1.0", tk.END)
        total = 0
        for item, qty in self.cart.items():
            price = prices.get(item, 0)
            line_total = price * qty
            self.bill_box.insert(tk.END, f"{item} x{qty} = {line_total:,} VND\n")
            total += line_total
        self.total_label.config(text=f"Tổng tiền: {total:,} VND")

# Hàm xóa giỏ hàng, tiếp tục tính tiền món ăn khác
    def clear_cart(self):
        self.cart = {}
        self.update_bill()
        self.freeze = False

# Hàm khởi tạo chạy chương trình (Tạo cửa sổ chính và khởi động GUI)
if __name__ == "__main__":
    play_background_music()
    app = App()
    app.mainloop()
    pygame.mixer.music.stop()
    pygame.mixer.quit()
