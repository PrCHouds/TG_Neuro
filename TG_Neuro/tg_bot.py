import PIL.Image
import telebot as tb
from main import *
bot = tb.TeleBot('6995134411:AAEGFcqbpeEAIp1XS4OljLCJYyIBh9jIFn4')
@bot.message_handler(content_types=['photo'])
def get_photo(message):
    file_info = bot.get_file(message.photo[len(message.photo)-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    src = 'photos/' + message.photo[1].file_id
    with open(src, 'wb') as new_file:
        new_file.write(downloaded_file)
    bot.reply_to(message, 'Фото загружено')
    img = PIL.Image.open(src)
    img.save('test.jpg')
    cartoon('test.jpg')
    bot.send_photo(message.chat.id, photo=open('results/test.jpg', 'rb'))
bot.polling()
