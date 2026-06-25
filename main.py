import sys
import os
import numpy as np
import cv2
import torch
from PIL import Image
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QTabWidget, QGraphicsScene, 
                             QGraphicsView, QFileDialog, QMessageBox, QRadioButton, QButtonGroup, QLabel)
from PyQt6.QtGui import QPixmap, QImage, QColor
from PyQt6.QtCore import Qt

# Импорт моделей
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

try:
    from simple_lama_inpainting import SimpleLama
except ImportError:
    SimpleLama = None

class ClickableScene(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_editor = parent
        self.image_item = None
        self.mask_item = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.image_item:
            pos = event.scenePos()
            x, y = int(pos.x()), int(pos.y())
            rect = self.image_item.pixmap().rect()
            if 0 <= x < rect.width() and 0 <= y < rect.height():
                self.parent_editor.handle_click(x, y)

class AILayerExtractor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI UI Layer Extractor (SAM 2 + Inpaint)")
        self.setGeometry(100, 100, 1200, 800)
        
        self.current_mask = None        
        self.cv_orig_image = None      
        self.extracted_qimage = None    
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Инициализация SAM 2
       # Инициализация SAM 2
        try:
            # Получаем абсолютный путь к текущей папке проекта
            base_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Строим абсолютные пути к файлам конфигурации и весов
            model_cfg = os.path.join(base_dir, "models", "sam2-hiera-tiny", "sam2_hiera_t.yaml")
            checkpoint = os.path.join(base_dir, "models", "sam2-hiera-tiny", "sam2_hiera_tiny.pt")
            
            print(f"Загрузка конфигурации: {model_cfg}")
            print(f"Загрузка весов: {checkpoint}")
            
            sam2_model = build_sam2(model_cfg, checkpoint, device=self.device)
            self.predictor = SAM2ImagePredictor(sam2_model)
            print(f"SAM 2 инициализирован на: {self.device}")
        except Exception as e:
            QMessageBox.warning(self, "Ошибка загрузки", f"Не удалось инициализировать SAM 2.\nОшибка: {e}")
            self.predictor = None

        # Инициализация LaMa
        self.lama = None
        if SimpleLama is not None:
            try:
                self.lama = SimpleLama()
                print("Модель LaMa Inpainting успешно готова к работе.")
            except Exception as e:
                print(f"Не удалось запустить LaMa: {e}")

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)

        top_panel = QHBoxLayout()
        self.btn_open = QPushButton("Открыть картинку")
        self.btn_open.clicked.connect(self.open_image)
        
        self.btn_extract = QPushButton("Extract")
        self.btn_extract.setEnabled(False)
        self.btn_extract.clicked.connect(self.extract_object)
        
        self.btn_save = QPushButton("Сохранить вырезанный объект")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save_extracted)

        top_panel.addWidget(self.btn_open)
        top_panel.addWidget(self.btn_extract)
        top_panel.addWidget(self.btn_save)
        
        top_panel.addSpacing(30)
        top_panel.addWidget(QLabel("Удаление фона:"))
        
        self.method_group = QButtonGroup(self)
        self.radio_opencv = QRadioButton("Быстрый (OpenCV)")
        self.radio_opencv.setChecked(True)
        self.method_group.addButton(self.radio_opencv)
        top_panel.addWidget(self.radio_opencv)
        
        self.radio_lama = QRadioButton("Качественный (ИИ LaMa)")
        if self.lama is None:
            self.radio_lama.setEnabled(False)
        self.method_group.addButton(self.radio_lama)
        top_panel.addWidget(self.radio_lama)

        top_panel.addStretch()
        main_layout.addLayout(top_panel)

        self.tabs = QTabWidget()
        self.scene_editor = ClickableScene(self)
        self.view_editor = QGraphicsView(self.scene_editor)
        self.tabs.addTab(self.view_editor, "Редактор (Исходник)")

        self.scene_result = QGraphicsScene(self)
        self.view_result = QGraphicsView(self.scene_result)
        self.tabs.addTab(self.view_result, "Вырезанный объект")

        main_layout.addWidget(self.tabs)
        self.setCentralWidget(main_widget)

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите скриншот", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.scene_editor.clear()
            self.scene_result.clear()
            
            self.cv_orig_image = cv2.imread(file_path)
            pixmap = QPixmap(file_path)
            self.scene_editor.image_item = self.scene_editor.addPixmap(pixmap)
            
            if self.predictor:
                rgb_image = cv2.cvtColor(self.cv_orig_image, cv2.COLOR_BGR2RGB)
                self.predictor.set_image(rgb_image)
            
            self.btn_extract.setEnabled(False)
            self.btn_save.setEnabled(False)
            self.tabs.setCurrentIndex(0)

    def handle_click(self, x, y):
        if self.cv_orig_image is None:
            return
            
        h, w, _ = self.cv_orig_image.shape

        if self.predictor:
            input_point = np.array([[x, y]])
            input_label = np.array([1])
            
            with torch.inference_mode():
                masks, scores, _ = self.predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=False
                )
            self.current_mask = masks[0]
        else:
            self.current_mask = np.zeros((h, w), dtype=bool)
            self.current_mask[max(0, y-50):min(h, y+50), max(0, x-70):min(w, x+70)] = True

        self.show_mask_overlay()
        self.btn_extract.setEnabled(True)

    def show_mask_overlay(self):
        if self.scene_editor.mask_item:
            self.scene_editor.removeItem(self.scene_editor.mask_item)
            
        h, w = self.current_mask.shape
        overlay_img = QImage(w, h, QImage.Format.Format_ARGB32)
        overlay_img.fill(QColor(0, 0, 0, 0))
        
        for y in range(h):
            for x in range(w):
                if self.current_mask[y, x]:
                    overlay_img.setPixelColor(x, y, QColor(0, 255, 150, 130))
                    
        mask_pixmap = QPixmap.fromImage(overlay_img)
        self.scene_editor.mask_item = self.scene_editor.addPixmap(mask_pixmap)

    def extract_object(self):
        if self.current_mask is None or self.cv_orig_image is None:
            return
            
        h, w, _ = self.cv_orig_image.shape
        
        # Шаг 1: Вырезание PNG слоя с альфа-каналом прозрачности
        rgba_image = cv2.cvtColor(self.cv_orig_image, cv2.COLOR_BGR2BGRA)
        rgba_image[~self.current_mask] = [0, 0, 0, 0]
        
        bytes_per_line = 4 * w
        self.extracted_qimage = QImage(rgba_image.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888).copy()
        
        self.scene_result.clear()
        self.scene_result.addPixmap(QPixmap.fromImage(self.extracted_qimage))
        
        # Шаг 2: Дорисовка фона (Inpainting)
        inpainting_mask = (self.current_mask.astype(np.uint8)) * 255
        
        if self.radio_opencv.isChecked():
            inpainted_bgr = cv2.inpaint(self.cv_orig_image, inpainting_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        else:
            img_rgb = cv2.cvtColor(self.cv_orig_image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            pil_mask = Image.fromarray(inpainting_mask)
            
            result_pil = self.lama(pil_img, pil_mask)
            inpainted_bgr = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)

        self.cv_orig_image = inpainted_bgr
        
        rgb_for_ui = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
        bytes_per_line_rgb = 3 * w
        updated_qimage = QImage(rgb_for_ui.data, w, h, bytes_per_line_rgb, QImage.Format.Format_RGB888)
        self.scene_editor.image_item.setPixmap(QPixmap.fromImage(updated_qimage))
        
        if self.predictor:
            self.predictor.set_image(rgb_for_ui)

        if self.scene_editor.mask_item:
            self.scene_editor.removeItem(self.scene_editor.mask_item)
            self.scene_editor.mask_item = None
            
        self.tabs.setCurrentIndex(1)
        self.btn_extract.setEnabled(False)
        self.btn_save.setEnabled(True)

    def save_extracted(self):
        if self.extracted_qimage is None:
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить слой", "layer.png", "PNG Image (*.png)")
        if file_path:
            self.extracted_qimage.save(file_path, "PNG")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AILayerExtractor()
    window.show()
    sys.exit(app.exec())
