import pickle
#产生数字dict
# f = open('./number_labels','wb')
# a = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9'}
# pickle.dump(a,f)
# f.close()
#
# x = open('./number_labels','rb')
# y = pickle.load(x)
# x.close()

#产生字母dict
f = open('./test_labels.txt','wb')
a = {0:'一',1:'二'}
pickle.dump(a,f,pickle.HIGHEST_PROTOCOL)
f.close()

x = open('./test_labels.txt','rb')
y = pickle.load(x)
x.close()





# import datetime
# dt = datetime.datetime(2018,11,20)
# print(dt.timestamp())
#
# from PIL import Image, ImageDraw, ImageFont
# im02 = Image.open("D:\\ img\\test02.jpg")
# draw = ImageDraw.Draw(im02)
# ft = ImageFont.truetype("C:\\WINDOWS\\Fonts\\SIMYOU.TTF", 20)
# draw.text((30,30), u"Python图像处理库PIL从入门到精通",font = ft, fill = 'red')
# ft = ImageFont.truetype("C:\\WINDOWS\\Fonts\\SIMYOU.TTF", 40)
# draw.text((30,100), u"Python图像处理库PIL从入门到精通",font = ft, fill = 'green')
# ft = ImageFont.truetype("C:\\WINDOWS\\Fonts\\SIMYOU.TTF", 60)
# draw.text((30,200), u"Python图像处理库PIL从入门到精通",font = ft, fill = 'blue')
# ft = ImageFont.truetype("C:\\WINDOWS\\Fonts\\SIMLI.TTF", 40)
# draw.text((30,300), u"Python图像处理库PIL从入门到精通",font = ft, fill = 'red')
# ft = ImageFont.truetype("C:\\WINDOWS\\Fonts\\STXINGKA.TTF", 40)
# draw.text((30,400), u"Python图像处理库PIL从入门到精通",font = ft, fill = 'yellow')
# im02.show()



