import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from openai import OpenAI
import matplotlib.pyplot as plt
import os
import logging
from logging.handlers import TimedRotatingFileHandler


def setup_logging():
    """
    Set up logger with file handler and console handler

    :return: logger
    """
    logger = logging.getLogger("telebot")
    logger.setLevel(logging.INFO)

    if not os.path.exists("logs"):
        os.makedirs("logs")
    handler = TimedRotatingFileHandler(
        "logs/telebot.log", when="D", interval=1, backupCount=5
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))

    # Add handler to the logger
    logger.addHandler(handler)

    # Add logging to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(console_handler)

    return logger


# Структура для хранение данных пользователей
user_data = {}

# API-ключи
TELEBOT_API_KEY = os.getenv('TELEBOT_API_KEY')
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai_client = OpenAI(api_key=OPENAI_API_KEY)
# Логирование
logger = setup_logging()


### Обработчики команд
async def start(update: Update, context):
    await update.message.reply_text("Добро пожаловать! Используйте /help для списка доступных команд.")


# Обработчик /help
async def help_command(update: Update, context):
    await update.message.reply_text(
        "Доступные команды:\n"
        "/start - Начало работы с ботом\n"
        "/set_profile - Настройка профиля (цели воды и калорий, запрос температуры)\n"
        "/log_water <количество> - Логировать количество выпитой воды (мл)\n"
        "/log_food <название продукта> - Логировать съеденный продукт (калорийность будет рассчитана)\n"
        "/log_workout <тип тренировки> <время (мин)> - Логировать тренировку и сожжённые калории\n"
        "/check_progress - Проверить текущий прогресс по воде и калориям\n"
        "/help - Показать список доступных команд"
    )


# Установка профиля
async def set_profile(update: Update, context):
    await update.message.reply_text("Как вас зовут?")
    context.user_data["step"] = "set_name"


# Обработчик ответов пользователя
async def handle_message(update: Update, context):
    user_id = update.message.from_user.id
    step = context.user_data.get("step")
    logger.info(f"handle_message. step={step}")

    if step == "set_name":
        username = update.message.text
        context.user_data["name"] = username
        await update.message.reply_text(f"Здравствуйте, {username}. Введите ваш рост (в см):")
        context.user_data["step"] = "set_height"

    elif step == "set_height":
        context.user_data["height"] = int(update.message.text)
        await update.message.reply_text("Введите ваш вес (в кг):")
        context.user_data["step"] = "set_weight"

    elif step == "set_weight":
        context.user_data["weight"] = float(update.message.text)
        await update.message.reply_text("Введите ваш возраст:")
        context.user_data["step"] = "set_age"

    elif step == "set_age":
        context.user_data["age"] = int(update.message.text)
        await update.message.reply_text("Введите ваш пол (м/ж):")
        context.user_data["step"] = "set_gender"

    elif step == "set_gender":
        gender = update.message.text.lower()
        if gender not in ["м", "ж"]:
            await update.message.reply_text("Необходимо указать 'м' или 'ж'.")
            return
        context.user_data["gender"] = gender
        await update.message.reply_text("В каком городе вы находитесь?")
        context.user_data["step"] = "set_city"

    elif step == 'set_city':
        user_input = update.message.text
        city = extract_city_from_text(user_input)
        if city:
            context.user_data["city"] = city
            await update.message.reply_text(f"Вы указали город: {city}.\n Сколько минут активности у вас в день?")
            context.user_data["step"] = "set_activity"
        else:
            await update.message.reply_text("Не удалось определить город. Попробуйте ещё раз.")

    elif step == "set_activity":
        context.user_data["activity_minutes"] = int(update.message.text)

        # Получение температуры через OpenWeatherMap
        city = context.user_data.get("city")
        # Получим температуру в городе с помощью OpenWeatherMap API
        temperature = get_city_temperature(city)
        context.user_data["temperature"] = temperature

        # Расчет цели по потреблению воды используя формулу
        water_goal = int(context.user_data["weight"] * 30 + 500 * context.user_data["activity_minutes"] / 30)
        if temperature > 25:
            water_goal += 500

        # Запоним структуру данных пользователя
        user_data[user_id] = {
            "city": context.user_data["city"],
            "height": context.user_data["height"],
            "weight": context.user_data["weight"],
            "age": context.user_data["age"],
            "gender": context.user_data["gender"],
            "activity_minutes": context.user_data["activity_minutes"],
            "temperature": temperature,
            "water": 0,
            "calories": 0,
            "calories_burned": 0,
            "water_goal": water_goal,
            "calories_goal": calculate_calories_goal(context.user_data),
        }

        # Покажем информацию профиля пользователю
        profile_info = (
            f"Ваш профиль создан:\n"
            f"Рост: {user_data[user_id]['height']} см\n"
            f"Вес: {user_data[user_id]['weight']} кг\n"
            f"Возраст: {user_data[user_id]['age']} лет\n"
            f"Пол: {'Мужской' if user_data[user_id]['gender'] == 'м' else 'Женский'}\n"
            f"Минуты активности: {user_data[user_id]['activity_minutes']} мин/день\n"
            f"Город: {user_data[user_id]['city']}\n"
            f"Температура в городе: {user_data[user_id]['temperature']}°C\n"
            f"Цель по калориям: {user_data[user_id]['calories_goal']} ккал/день\n"
            f"Цель по воде: {user_data[user_id]['water_goal']} мл/день"
        )
        await update.message.reply_text(profile_info)
        # Окончили обработку ввода данных по профилю, очистим step
        del context.user_data["step"]

    elif step == "log_food":
        weight_in_grams = int(update.message.text)
        product_name = context.user_data["product_name"]
        #
        logger.info(f"weight_in_grams={weight_in_grams}, product_name={product_name}")
        # Вызовем функцию рассчета калорий с помощью LLM
        calories = calculate_food_calories(product_name, weight_in_grams)

        user_id = update.message.from_user.id
        user_data[user_id]["calories"] += calories

        txt = f"Записано: {calories:.1f} ккал из продукта '{product_name}'."
        await update.message.reply_text(txt)

        del context.user_data["step"]


def extract_city_from_text(user_input):
    """
    Использует OpenAI для извлечения названия города из текста.

    :param user_input: str
    :return: str
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Ты хорошо разбираешься в географии и знаешь названия всех городов"},
                {"role": "user",
                 "content": f"Определи город о котором идет речь в тексте: '{user_input}'. Верни только официальное название города без любых других символов. Если не удается определить город, напиши 'None'."}
            ],
            max_tokens=50
        )
        city = response.choices[0].message.content.strip()[:255]
        return city if city.lower() != "none" else None
    except Exception as e:
        logger.info(f"Ошибка при запросе к OpenAI: {e}")
        return None


def get_city_temperature(city):
    """
    Получение текущей температуры в городе с использованием OpenWeatherMap API

    :param city: str
    :return: float
    """
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()
        if response.status_code == 200 and "main" in data:
            return data["main"]["temp"]
        else:
            logger.info(f"Ошибка OpenWeatherMap: {data.get('message', 'Неизвестная ошибка')}")
            return None
    except Exception as e:
        logger.info(f"Ошибка при запросе температуры: {e}")
        return None


def calculate_food_calories(product_name, weight_in_grams):
    """
    Определение калорийности продукта при помощи OpenAI API

    :param product_name: str
    :param weight_in_grams: int
    :return: float
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Ты диетолог со знанием калорийности продуктов"},
                {"role": "user",
                 "content": f"Сколько калорий в {weight_in_grams} граммах продукта '{product_name}'? В ответе укажи только число. Если не удалось определить, то укажи 0"}
            ],
            max_tokens=50
        )
        calories_info = response.choices[0].message.content.strip()[:255]
        #
        logger.info(f"Ответ GPT по калорийности {product_name}: {calories_info}")
        #
        calories = float(calories_info.split()[0])
    except Exception as e:
        logger.info(f"Не удалось определить количество калорий в {weight_in_grams} граммах продукта '{product_name}: {e}")
        #
        calories = 0
    return calories


def calculate_calories_goal(user_data):
    """
    Рассчитывает базовую норму калорий на основе данных пользователя.

    :param user_data: str
    :return: int
    """

    weight = user_data["weight"]
    height = user_data["height"]
    age = user_data["age"]
    gender = user_data["gender"]
    activity_minutes = user_data["activity_minutes"]

    # Формула Mifflin-St Jeor
    if gender == "м":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    activity_factor = 1.2 + (activity_minutes / 60) * 0.05  # умеренный активный образ жизни
    return int(bmr * activity_factor)


async def log_water(update: Update, context):
    user_id = update.message.from_user.id
    if len(context.args) != 1 or not context.args[0].isdigit():
        await update.message.reply_text("Используйте: /log_water <количество> мл")
        return

    amount = int(context.args[0])
    user_data[user_id]["water"] += amount
    water_left = max(0, user_data[user_id]["water_goal"] - user_data[user_id]["water"])
    await update.message.reply_text(f"Выпито: {user_data[user_id]['water']} мл. Осталось: {water_left} мл.")



async def log_food(update: Update, context):
    if len(context.args) < 1:
        await update.message.reply_text("Используйте: /log_food <название продукта>")
        return

    product_name = " ".join(context.args)
    await update.message.reply_text(f"Введите вес продукта '{product_name}' в граммах:")
    context.user_data["step"] = "log_food"
    context.user_data["product_name"] = product_name


async def log_workout(update: Update, context):
    if len(context.args) < 2:
        await update.message.reply_text("Используйте: /log_workout <тип тренировки> <время (мин)>")
        return

    workout_type = context.args[0].lower()
    time = int(context.args[1])

    calories_burned = time * 10  # Условный расход калорий
    additional_water = (time // 30) * 200

    user_id = update.message.from_user.id
    user_data[user_id]["calories_burned"] += calories_burned
    user_data[user_id]["water"] += additional_water

    await update.message.reply_text(
        f"{workout_type.capitalize()} {time} минут — {calories_burned} ккал. "
        f"Дополнительно: выпейте {additional_water} мл воды."
    )


async def generate_progress_pie_chart(water_consumed, water_goal, calories_consumed, calories_goal):
    """
    Генерирует круговую диаграмму прогресса по воде и калориям.

    :param water_consumed: int
    :param water_goal: int
    :param calories_consumed: int
    :param calories_goal: int
    :return: str
    """

    # Данные для воды
    water_percentage = min(water_consumed / water_goal, 1.0) * 100
    water_remaining = 100 - water_percentage

    # Данные для калорий
    calories_percentage = min(calories_consumed / calories_goal, 1.0) * 100
    calories_remaining = 100 - calories_percentage

    # Создание круговых диаграмм
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Диаграмма воды
    ax1.pie([water_percentage, water_remaining], labels=['Потреблено', 'Осталось'],
            autopct='%1.1f%%', startangle=90, explode=(0.1, 0), pctdistance=0.85)
    ax1.set_title("Прогресс по воде")

    # Диаграмма калорий
    ax2.pie([calories_percentage, calories_remaining], labels=['Потреблено', 'Осталось'],
            autopct='%1.1f%%', startangle=90, explode=(0.1, 0), pctdistance=0.85)
    ax2.set_title("Прогресс по калориям")

    # Сохранение диаграммы в файл
    output_path = "progress_pie_chart.png"
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


async def check_progress(update: Update, context):
    """
    Вывод прогресса пользователя
    """

    user_id = update.message.from_user.id
    user = user_data.get(user_id,
                         {"water": 0, "calories": 0, "calories_burned": 0, "water_goal": 2400, "calories_goal": 2500})

    water_left = max(0, user["water_goal"] - user["water"])
    calorie_balance = user["calories"] - user["calories_burned"]

    await update.message.reply_text(
        f"Прогресс:\n"
        f"Вода:\n- Выпито: {user['water']} мл из {user['water_goal']} мл.\n- Осталось: {water_left} мл.\n\n"
        f"Калории:\n- Потреблено: {user['calories']} ккал из {user['calories_goal']} ккал.\n"
        f"- Сожжено: {user['calories_burned']} ккал.\n- Баланс: {calorie_balance} ккал."
    )

    # Генерация диаграммы
    chart_path = await generate_progress_pie_chart(
        water_consumed=user["water"],
        water_goal=user["water_goal"],
        calories_consumed=user["calories"],
        calories_goal=user["calories_goal"]
    )

    # Отправка диаграммы пользователю
    with open(chart_path, "rb") as chart_file:
        await context.bot.send_photo(chat_id=update.effective_chat.id, photo=chart_file)


# Главная функция приложения
def main():
    app = Application.builder().token(TELEBOT_API_KEY).build()

    # Обработчики команд
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("set_profile", set_profile))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CommandHandler("log_water", log_water))
    app.add_handler(CommandHandler("log_food", log_food))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_food_weight))
    app.add_handler(CommandHandler("log_workout", log_workout))
    app.add_handler(CommandHandler("check_progress", check_progress))

    # Запуск бота
    print("Бот запущен...")
    app.run_polling()


if __name__ == "__main__":
    main()
