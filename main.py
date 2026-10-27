import os
import uuid
import cv2
import easyocr
import numpy as np
import telebot
from dotenv import load_dotenv

# Загружаем переменные из файла .env
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")

# Инициализируем бота по защищенному токену
bot = telebot.TeleBot(BOT_TOKEN)
reader = easyocr.Reader(["ru", "en"])

UPLOAD_DIR = "images"
os.makedirs(UPLOAD_DIR, exist_ok=True)



def process_and_ocr(image_path):
    """Функция сканирования и извлечения текста из фото"""
    image = cv2.imread(image_path)
    if image is None:
        return "Ошибка: Не удалось прочитать изображение."

    orig = image.copy()

    # 1. Предобработка для поиска контуров
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 75, 200)

    contours, _ = cv2.findContours(
        edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    doc_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            doc_contour = approx
            break

    # 2. Выравнивание перспективы (трансформация)
    if doc_contour is not None:
        pts = doc_contour.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(heightA), int(heightB))

        # ОШИБКА ПОЛНОСТЬЮ ИСПРАВЛЕНА ЗДЕСЬ:
        dst = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1]
            ],
            dtype="float32"
        )

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(orig, M, (max_width, max_height))
        scan_result = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    else:
        # Если ровные границы листа не найдены, используем оригинальный ч/б вариант
        scan_result = gray

    # Повышаем контрастность (бинаризация)
    scan_result = cv2.threshold(
        scan_result, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )[1]

    # Перезаписываем файл уже в отсканированном виде
    cv2.imwrite(image_path, scan_result)

    # 3. Распознавание текста
    result = reader.readtext(scan_result, detail=0)
    if not result:
        return "Текст на изображении не обнаружен."

    return "\n".join(result)


@bot.message_handler(commands=["start"])
def send_welcome(message):
    bot.reply_to(
        message,
        "Привет! Я твой Telegram бот-сканер. 📄\nПросто отправь мне фото документа, и я распознаю текст!",
    )


@bot.message_handler(commands=["hello"])
def send_hello(message):
    bot.reply_to(
        message,
        "Привет! Как дела? Жду твоего фото, чтобы превратить его в текст. 😊",
    )


@bot.message_handler(content_types=["photo"])
def handle_photo(message):
    try:
        
        status_msg = bot.reply_to(message, "Сканирую документ, подождите...")

        photo = message.photo[-1]
        file_info = bot.get_file(photo.file_id)
        extension = os.path.splitext(file_info.file_path)[-1] or ".jpg"
        file_name = f"{uuid.uuid4().hex}{extension}"
        file_path = os.path.join(UPLOAD_DIR, file_name)

        
        downloaded_file = bot.download_file(file_info.file_path)
        with open(file_path, "wb") as new_file:
            new_file.write(downloaded_file)

        
        recognized_text = process_and_ocr(file_path)

        
        bot.delete_message(message.chat.id, status_msg.message_id)

        
        if len(recognized_text) > 4000:
            for x in range(0, len(recognized_text), 4000):
                bot.send_message(message.chat.id, recognized_text[x : x + 4000])
        else:
            bot.send_message(message.chat.id, recognized_text)

    except Exception as error:
        print(f"Произошла ошибка: {error}")
        bot.reply_to(message, f"Произошла ошибка при обработке фото: {error}")



bot.polling()
