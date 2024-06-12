import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import torch
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDHead, det_utils
from torchvision.transforms.functional import to_tensor
import os

# Завантаження ваг моделі та ініціалізація моделі
model_weights_pth = "E:/Code/Diploma/model.pth"
model = ssd300_vgg16()
device = "cpu"
in_channels = det_utils.retrieve_out_channels(model.backbone, (480, 480))
num_anchors = model.anchor_generator.num_anchors_per_location()
model.head = SSDHead(in_channels=in_channels, num_anchors=num_anchors, num_classes=2)
model.load_state_dict(torch.load(model_weights_pth, map_location=device))
model.to(device)
model.eval()

# Глобальні змінні для зображення
img = None
img_tk = None

# Функція для завантаження зображення
def load_image():
    global img, img_tk
    file_path = filedialog.askopenfilename()  # Відкриває вікно вибору файлу та повертає шлях обраного файлу
    if file_path:
        try:
            img = Image.open(file_path).convert("RGB")  # Відкриває та конвертує зображення у RGB формат
            img = img.resize((480, 480))  # Змінює розмір зображення на 480x480
            img_tk = ImageTk.PhotoImage(img)  # Конвертує зображення для відображення в Tkinter
            panel.imgtk = img_tk  # Оновлює зображення в панелі
            panel.config(image=img_tk)  # Конфігурує зображення в панелі
            analyze_btn.config(state=tk.NORMAL)  # Активує кнопку "Analyze Image"

            # Відображення назви обраного зображення
            image_name = os.path.basename(file_path)  # Отримання лише імені файлу без шляху
            image_name_label.config(text="Назва зображення: " + image_name)  # Встановлює текст для мітки
            mines_label.config(text=" ")
            error_label.config(text="")  # Очищуємо повідомлення про помилку

        except Exception as e:
            messagebox.showerror("Помилка", f"Не вдалося завантажити зображення: {e}")
            error_label.config(text=f"Не вдалося завантажити зображення: {e}")
            analyze_btn.config(state=tk.DISABLED)  # Вимикає кнопку "Analyze Image" у разі помилки


# Функція для аналізу зображення
def analyze_image():
    global img, img_tk
    if img: 
        confidence_threshold = confidence_scale.get() / 100.
        img_tensor = to_tensor(img).unsqueeze(0).to(device)  # Перетворюємо зображення у тензор та переміщуємо його на пристрій
        
        with torch.no_grad():  # Вимикаємо обчислення градієнтів для швидшого виконання
            output = model(img_tensor)  # Передаємо зображення моделі для отримання виходу
        
        boxes = output[0]['boxes'].cpu().numpy()  # Отримуємо координати областей обмежень
        scores = output[0]['scores'].cpu().numpy()  # Отримуємо бали впевненості
        labels = output[0]['labels'].cpu().numpy()  # Отримуємо мітки класів
        
        high_conf_indices = scores >= confidence_threshold  # Індекси елементів з високим балом впевненості
        boxes = boxes[high_conf_indices]  # Фільтруємо області обмежень за високим балом впевненості
        scores = scores[high_conf_indices]  # Фільтруємо бали впевненості за високим балом впевненості
        labels = labels[high_conf_indices]  # Фільтруємо мітки класів за високим балом впевненості

        num_mines = sum(labels == 1)

        # Відображаємо кількість мін або повідомлення про їх відсутність
        if num_mines > 0:
            mines_label.config(text=f"Знайдено мін: {num_mines}")
        else:
            mines_label.config(text="Мін не знайдено")

        # Копіюємо зображення для малювання боксів
        img_with_boxes = img.copy()
        draw = ImageDraw.Draw(img_with_boxes)
        
        # Проходимося по кожному боксу та малюємо його на зображенні
        for box, score, label in zip(boxes, scores, labels):
            xmin, ymin, xmax, ymax = box
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=2)  # Малюємо прямокутник
            draw.text((xmin, ymin), f'{label}: {score:.2f}', fill="red")  # Додаємо текст з міткою та балом впевненості
        
        # Оновлюємо відображене зображення в вікні Tkinter з намальованими боксами
        img_tk = ImageTk.PhotoImage(img_with_boxes)
        panel.imgtk = img_tk
        panel.config(image=img_tk)

def show_project_info():
    project_info_window = tk.Toplevel(root)
    project_info_window.title("Дипломний проєкт")
    project_info_window.geometry("450x350")
    
    info_text_1 = """
    Дипломний проєкт
    здобуття ступеня бакалавра
    за освітньо-професійною програмою «Інформаційне забезпечення 
    робототехнічних систем»
    спеціальності 126 «Інформаційні системи та технології»
    на тему: «Роботизована система для виявлення мінних полів»
    """

    info_text_2 = """

    Виконав: 
    студент IV курсу, групи ІК-03 
    Цвілій Богдан Олександрович

    Керівник: 
    Асистент кафедри ІСТ,
    Мягкий Михайло Юрійович
    """
    info_label_1 = tk.Label(project_info_window, text=info_text_1, justify="center")
    info_label_1.pack(pady=10)

    info_label_2 = tk.Label(project_info_window, text=info_text_2, justify="left")
    info_label_2.pack(pady=10, side='left')

        # Функція для закриття вікна інформації про проект
    def close_project_info():
        project_info_window.destroy()
        root.deiconify()  # Повернення до головного вікна програми
    
    close_btn = tk.Button(project_info_window, text="Далі", command=close_project_info)
    close_btn.pack(pady=10, side='bottom')

    root.withdraw()  # Ховаємо головне вікно програми

# Інтерфейс Tkinter
root = tk.Tk()
root.title("Виявлення мін")
root.geometry("800x600")

show_project_info()

# Панель для зображення
panel = tk.Label(root)
panel.grid(row=1, column=0, padx=10, pady=10)

# Фрейм для кнопок та поля введення
btn_frame = tk.Frame(root)
btn_frame.grid(row=1, column=1, padx=10, pady=10, sticky="n")

# Фрейм для назви зображення
title_frame = tk.Frame(root)
title_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

# Елементи у фреймі назви зображення
image_name_label = tk.Label(title_frame, text="Назва зображення: ")
image_name_label.pack()

# Елементи у фреймі кнопок та поля введення
load_btn = tk.Button(btn_frame, text="Завантажити зображення", command=load_image)
load_btn.grid(row=0, column=0, pady=5)

confidence_label = tk.Label(btn_frame, text="Поріг впевненості:")
confidence_label.grid(row=1, column=0, pady=5)

confidence_scale = tk.Scale(btn_frame, from_=1, to=100, orient=tk.HORIZONTAL)
confidence_scale.set(50)  # Початкове значення
confidence_scale.grid(row=1, column=1, pady=5)

analyze_btn = tk.Button(btn_frame, text="Проаналізувати зображення", command=analyze_image, state=tk.DISABLED)
analyze_btn.grid(row=2, column=0, pady=5)

mines_label = tk.Label(btn_frame, text="")
mines_label.grid(row=3, column=0)

error_label = tk.Label(btn_frame, text="", fg="red")
error_label.grid(row=4, column=0)

root.mainloop()  # Запускаємо головний цикл Tkinter для відображення вікна