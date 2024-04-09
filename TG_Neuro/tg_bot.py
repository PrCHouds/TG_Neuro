import PIL.Image
import telebot as tb
from multi import *
# new telebot cat
bot = tb.TeleBot('6995134411:AAEGFcqbpeEAIp1XS4OljLCJYyIBh9jIFn4')
s = 0
@bot.message_handler(content_types=['text'])
def start_massage(message):
    if message.text == 'help':
        bot.send_message(message.chat.id, "Давай знакомиться)\n"
                        "Если ты любишь котиков так же, как и я, то тебе точно понравится, меняя изображения с ними, делать этот мир чуточку прекраснее\n"
                        "Да и даже если не любишь котиков, менять изображения тебе понравится!\n"
                         "\n""Я смог заинтересовать тебя?\n"
                         "Тогда выбирай изображение И СРАЗУ ВМЕСТЕ С НИМ ПИШИ КОМАНДУ, как будем менять изображение ее помощью!\n"
                         "multi - сделаем из твоей картинки мультяшку\n"
                         "gray - делаем старую картинку  :3")
    elif message.text == 'multi':
        bot.send_message(message.chat.id, "Значит давай делать мультяшку. Закидывай фотографию")
        s = 0
        get_photo(message)
    else:
        bot.send_message(message.chat.id, "Привет, юный любитель котиков, чтобы начать работу напиши мне 'help' ")



@bot.message_handler(content_types=['photo', 'text'])
def get_photo(message):
    if s == 0:
        if message.photo:
            file_info = bot.get_file(message.photo[len(message.photo)-1].file_id)
            downloaded_file = bot.download_file(file_info.file_path)
            src = 'photos/' + message.photo[1].file_id
            with open(src, 'wb') as new_file:
                new_file.write(downloaded_file)
            bot.reply_to(message, 'Фото загружено. Делаем мультяшку...')
            img = PIL.Image.open(src)
            img.save('test.jpg')
            cartoon('test.jpg')
            bot.send_photo(message.chat.id, photo=open('results/test.jpg', 'rb'))
bot.polling()
