"""Рисует историю на Python. Из переданного MP4 используется только звук.

Установка: python -m pip install pillow numpy imageio-ffmpeg
Запуск:    python draw_animation.py VID_20261127_073147_828.mp4
"""
import argparse
import math
import random
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

S = 3
W, H = 320, 180
FPS, FRAMES = 30, 979
SIZE = (W * S, H * S)
INK, EDGE = "#19121f", "#ad95a1"
BLUE, RED = "#568beb", "#dc465f"


def smooth(t):
    t = min(1, max(0, t))
    return t * t * (3 - 2 * t)


def ramp(t, start, end):
    return smooth((t - start) / (end - start))


def xy(points):
    return [(round(x * S), round(y * S)) for x, y in points]


def line(d, points, color, width=1):
    d.line(xy(points), fill=color, width=max(1, round(width * S)), joint="curve")


def poly(d, points, color):
    d.polygon(xy(points), fill=color)


def ellipse(d, x, y, rx, ry, color, outline=None, width=0.5):
    d.ellipse((round((x-rx)*S), round((y-ry)*S), round((x+rx)*S), round((y+ry)*S)),
              fill=color, outline=outline, width=max(1, round(width*S)))


def rect(d, x1, y1, x2, y2, color):
    d.rectangle((round(x1*S), round(y1*S), round(x2*S), round(y2*S)), fill=color)


def heart(d, x, y, size, color):
    points = []
    for i in range(64):
        u = math.tau * i / 64
        points.append((x + size * math.sin(u)**3,
                       y - size * (13*math.cos(u)-5*math.cos(2*u)-2*math.cos(3*u)-math.cos(4*u))/16))
    poly(d, points, color)


def font(size):
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf",
                 "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                 "/system/fonts/Roboto-Bold.ttf"):
        try:
            return ImageFont.truetype(name, round(size * S))
        except OSError:
            pass
    return ImageFont.load_default(size=round(size * S))


CAPTION_FONT = font(9)
MARK_FONT = font(5)
BG_Y, BG_X = np.mgrid[0:360, 0:640].astype(np.float32) / 2
SCENES = [
    (0.00, "heart"), (2.10, "umbrella"), (3.10, "window"), (5.85, "rain"),
    (7.88, "cinema"), (10.10, "sunset"), (12.29, "walk"), (14.13, "hills"),
    (15.65, "water"), (17.08, "hands"), (20.47, "bench"), (22.84, "autumn"),
    (23.40, "snow"), (24.20, "evening"), (24.93, "spring"), (26.40, "together"),
    (863 / FPS, "credits"),
]
# Красное выделение обозначено звёздочками.
CAPTIONS = [
    (.37, 2.04, 'Может быть, я не умею *любить,*'),
    (2.16, 3.06, 'дарить *заботу,*'),
    (3.17, 5.79, '*тепло,* как это делают все,'),
    (5.90, 7.79, 'но я такой, какой я *есть.*'),
    (7.97, 9.97, 'Да, не *идеальный* парень из фильма,'),
    (10.13, 12.07, 'но зато *живой* и *настоящий.*'),
    (12.4, 14.01, 'И я знаю *точно,*'),
    (14.16, 15.50, 'что ты мне *нужна.*'),
    (15.65, 16.94, 'Нужна как *кислород,*'),
    (17.16, 20.01, 'без которого жить просто *невозможно.*'),
    (20.63, 22.76, 'И если мне потребуется *ждать* тебя,'),
    (22.90, 24.81, 'я готов ждать *столько,*'),
    (24.90, 26.37, 'сколько *потребуется.*'),
]


def sky(top, bottom, light=None, rays=False):
    y = np.minimum(BG_Y / 150, 1)[:, :, None]
    field = np.array(top) * (1-y) + np.array(bottom) * y
    if light:
        x, h, intensity = light
        radius = ((BG_X-x)/100)**2 + ((BG_Y-h)/91)**2
        glow = np.exp(-1.7*radius) * intensity
        if rays:
            angle = np.arctan2(BG_X-x, h-BG_Y)
            glow += 10 * (0.5+0.5*np.cos(22*angle))**5 * np.exp(-radius) * (BG_Y<h)
        field += glow[:, :, None] * np.array((1, .79, .57))
    vignette = np.minimum(1, ((BG_X-160)/210)**2+((BG_Y-85)/200)**2)
    field -= 18*vignette[:, :, None]
    grain = np.random.default_rng(2026).normal(0, .9, BG_X.shape)
    field += grain[:, :, None]
    return Image.fromarray(np.uint8(np.clip(field, 0, 255))).resize(SIZE, Image.Resampling.BILINEAR)


def hill(d, level, amplitude, phase, color):
    pts = [(x, level + amplitude*math.sin(x/69+phase) + amplitude*.3*math.sin(x/29+.3))
           for x in range(-2, W+4, 2)]
    poly(d, pts + [(W+2, H), (-2, H)], color)


def tree(d, season):
    rng = random.Random(209)
    dark = '#382b3a'
    def branch(x, y, angle, length, width, depth):
        xx, yy = x+length*math.cos(angle), y+length*math.sin(angle)
        line(d, [(x,y),(xx,yy)], dark, width)
        if depth:
            branch(xx, yy, angle-rng.uniform(.23,.65), length*.68, width*.64, depth-1)
            branch(xx, yy, angle+rng.uniform(.15,.56), length*.70, width*.64, depth-1)
        elif season in ('autumn','spring'):
            colors = ('#ca834a','#c07a48','#d69352') if season=='autumn' else ('#e7a6bc','#f3b7ca','#e99ab3')
            for _ in range(8):
                ellipse(d, xx+rng.uniform(-4,4), yy+rng.uniform(-4,4), 2.4, 2.3, rng.choice(colors))
    line(d, [(56,157),(58,118),(52,91)], dark, 3.5)
    for x,y,angle,length in [(56,122,-2.06,33),(53,97,-1.90,35),(54,100,-.96,39),(52,94,-2.42,27)]:
        branch(x,y,angle,length,2.6,4)


def bench(d, lamp=True):
    metal = '#282333'
    for x in (120,205):
        line(d,[(x,107),(x,157)],metal,1.3)
    line(d,[(118,107),(208,107)],metal,1.6)
    line(d,[(118,128),(208,128)],metal,2.3)
    if lamp:
        line(d,[(215,46),(215,159)],'#38313d',1)
        ellipse(d,215,46,1.5,2,'#ffe0ab')


@lru_cache(maxsize=24)
def background(scene):
    night = scene in ('heart','umbrella','window','rain','bench','cinema')
    wet = scene in ('snow','evening','water')
    if night:
        im = sky((28,27,51),(56,47,78),(215,50,37) if scene in ('umbrella','bench') else None)
    elif scene=='water':
        im = sky((41,70,108),(7,15,36),(150,-44,18))
    elif wet:
        palette = ((93,106,139),(186,191,208)) if scene=='snow' else ((55,69,96),(134,129,151))
        im = sky(*palette,(215,43,25))
    elif scene=='autumn':
        im = sky((118,79,107),(238,177,127))
    else:
        sun = 160 if scene in ('hands','together') else 239 if scene in ('sunset','hills') else 263 if scene=='walk' else 281
        im = sky((99,52,103),(247,135,131),(sun,123,64),True)
    d=ImageDraw.Draw(im)
    if night:
        rng=random.Random(32)
        for i in range(18):
            x=i*19+rng.randrange(-4,4)
            top=rng.randrange(83,126)
            rect(d,x,top,x+rng.randrange(11,22),154,rng.choice(('#28273e','#2c2a45','#302d47')))
            for _ in range(4):
                xx=x+rng.randint(2,12); yy=rng.randint(top+4,146)
                rect(d,xx,yy,xx+1.1,yy+2,'#3c354d')
        rect(d,0,150,W,H,'#141321')
    if scene in ('heart','window'):
        left=203 if scene=='heart' else 124
        rect(d,left,7,W,159,'#14121e')
        if scene=='heart':
            rect(d,239,32,271,66,'#e2bc83')
            line(d,[(255,31),(255,67)],'#322733',2)
        else:
            rect(d,159,33,276,110,'#a5653a')
            line(d,[(218,32),(218,111)],'#3e282b',1.3)
            rect(d,166,91,217,104,'#392124')
            rect(d,242,77,272,110,'#39232a')
            ellipse(d,255,102,6,6,'#e0ba7a')
            for i in range(2):
                x=183+8*i
                ellipse(d,x,70+i*4,3.4,4,'#36232a')
                line(d,[(x,75+i*4),(x+2,89),(x+9,92),(x+11,104)],'#352129',3.6)
            line(d,[(156,114),(279,114)],'#d9ccba',1.6)
            rect(d,0,158,W,H,'#686674')
    elif scene=='cinema':
        im=sky((35,25,38),(17,16,27))
        d=ImageDraw.Draw(im)
        panel=background('together').resize((240*S,86*S),Image.Resampling.LANCZOS)
        im.paste(panel,(40*S,11*S))
        for x in range(0,320,23):
            rect(d,x,136,x+17,176,'#0e0c15')
    elif scene=='water':
        for x in (25,86,163,230):
            poly(d,[(x,0),(x+6,0),(x+15,H),(x-8,H)],'#263654')
    elif not night:
        if scene not in ('snow','evening','autumn'):
            sx=160 if scene in ('hands','together') else 239 if scene in ('sunset','hills') else 263 if scene=='walk' else 281
            ellipse(d,sx,121,12,12,'#fff0d4')
        if scene!='together':
            hill(d,129,9,.65,'#927194')
            hill(d,143,8,2.10,'#604768')
            hill(d,149,7,3.92,'#292137' if not wet else '#62596b')
        else:
            rect(d,0,167,320,180,'#281b29')
    if scene in ('autumn','snow','evening','spring'):
        tree(d,scene)
        bench(d,lamp=scene!='spring')
    if scene in ('umbrella','bench'):
        if scene=='bench': bench(d)
        else:
            line(d,[(269,52),(269,159)],'#403847',1)
            ellipse(d,269,50,1.6,1.6,'#ffe0ab')
    return im


def person(d, x, foot, h, girl=False, facing=1, reach=0, step=0, seated=False, underwater=False, lift=0):
    # Локальные координаты скелета: ступни на уровне 0, макушка на уровне 1.
    def p(a,b): return x+facing*h*a, foot-b*h
    def limb(points,width):
        pts=[p(a,b) for a,b in points]
        line(d,pts,EDGE,h*width+1.1)
        line(d,pts,INK,h*width)
    if seated:
        limb([(0,.49),(-.025,.33),(.27,.33),(.30,.025),(.42,.018)],.073)
        limb([(-.016,.78),(0,.54),(.23,.415)],.052)
    else:
        for sign in (-1,1):
            stride = sign*step*.19
            limb([(0,.46),(stride*.48,.23),(stride,.025),(stride+.071,.016)],.058)
    if girl:
        poly(d,[p(-.043,.78),p(.048,.78),p(.041,.63),p(.102,.23),p(-.104,.23),p(-.037,.62)],INK)
    else:
        poly(d,[p(-.064,.80),p(.067,.80),p(.042,.435),p(-.052,.435)],INK)
    if not seated:
        if underwater:
            limb([(.025,.75),(.12,.83),(.19,.98)],.043)
            limb([(-.025,.74),(.065,.87),(.10,.99)],.043)
        else:
            limb([(.03,.75),(.045+reach*.100,.61+lift*.45),
                  (.045+reach*.22+step*.043,.49+reach*.093+lift)],.05)
    qx,qy=p(0,.891)
    ellipse(d,qx,qy,.069*h,.074*h,EDGE)
    ellipse(d,qx-.005*h,qy+.003*h,.067*h,.071*h,INK)
    if girl:
        poly(d,[p(-.057,.923),p(.034,.945),p(.075,.862),p(.077,.737),p(-.073,.742)],INK)
    scarf=RED if girl else BLUE
    poly(d,[p(-.049,.812),p(.050,.812),p(.058,.773),p(-.048,.773)],scarf)
    poly(d,[p(.026,.785),p(.057,.786),p(.043,.685),p(.022,.694)],scarf)


def weather(d, t, snow=False):
    rng=random.Random(77 if snow else 42)
    for _ in range(112 if snow else 142):
        x, y = rng.random()*330, rng.random()*198
        speed = rng.uniform(12,32) if snow else rng.uniform(85,140)
        xx=(x+t*(1.2 if snow else 6.5))%330-5
        yy=(y+t*speed)%194-7
        if snow:
            r=rng.uniform(.24,.76)
            ellipse(d,xx,yy,r,r,'#b9b8c6')
        else:
            line(d,[(xx,yy),(xx+.55,yy+3+rng.random()*3)],'#444057',.27)


def instagram(d,x,y,size,color):
    a,b=round(x*S),round(y*S)
    side=round(size*S)
    d.rounded_rectangle((a,b,a+side,b+side),radius=round(side*.27),outline=color,width=max(1,round(side*.085)))
    ellipse(d,x+size/2,y+size/2,size*.228,size*.228,None,color,size*.079)
    ellipse(d,x+size*.76,y+size*.24,size*.055,size*.055,color)


def mark(d,x,y):
    instagram(d,x,y,14,'#d3ced4')
    d.text((round((x+7)*S),round((y+24)*S)),'@SAGE.ZF',font=MARK_FONT,anchor='mt',fill='#e4dce1')


def shot(scene,t):
    if scene=='credits':
        im=Image.blend(Image.new('RGB',SIZE),shot('together',28.6),.11)
        d=ImageDraw.Draw(im)
        pulse=ramp(t,29.25,29.62)*(1-ramp(t,31.52,32.05))
        color=tuple(round(a+(b-a)*pulse) for a,b in zip((231,221,226),(225,15,143)))
        instagram(d,141,42,38,color)
        if t>=869/FPS: d.text((160*S,96*S),'@SAGE.ZF',font=font(8),anchor='mt',fill='#e2dbe0')
        return im
    im=background(scene).copy()
    d=ImageDraw.Draw(im)
    if scene=='heart':
        reach=1-ramp(t,1.4,1.72)
        person(d,147,164,124,reach=reach*1.15,lift=.16*reach)
        if reach>.02: heart(d,151+35*reach,96-29*reach,8*reach,'#ef405d')
    elif scene=='umbrella':
        person(d,139,164,121,reach=.75,lift=.28)
        person(d,180,164,115,True,-1)
        open_by=ramp(t,2.2,2.55)
        x,y=167,24-20*open_by
        line(d,[(164,63),(x,y+5)],'#9c8d98',.8)
        poly(d,[(x+(i-15)*2.25*open_by,y+18*((i-15)/15)**2) for i in range(31)],'#17151e')
        weather(d,t)
    elif scene in ('window','rain'):
        person(d,68 if scene=='window' else 159,169,128,reach=.55)
        weather(d,t,scene=='window')
    elif scene=='cinema':
        person(d,140,101,57,reach=1,seated=True)
        person(d,181,101,64,True,-1,reach=.55)
        heart(d,160,39,4,'#ec6687')
        person(d,168,177,54,seated=True)
    elif scene in ('sunset','walk'):
        walk=ramp(t,12.73,13.37) if scene=='walk' else 0
        x=125+26*walk if scene=='walk' else 143
        person(d,x,179 if scene=='walk' else 164,136 if scene=='walk' else 127,
               step=math.sin(math.pi*walk))
    elif scene=='hills':
        person(d,75,140,76)
        person(d,240,140,73,True,-1)
    elif scene=='water':
        person(d,158,190,97,underwater=True)
        for i in range(20):
            yy=(103-i*7-t*19)%120-13
            xx=159+7*math.sin(yy/12+i*.45)
            ellipse(d,xx,yy,1+i%3*.39,1+i%3*.39,None,'#64758c',.23)
    elif scene in ('hands','together'):
        time=17.35 if scene=='hands' else 26.49
        extend=ramp(t,time,time+.32)
        person(d,136,164 if scene=='hands' else 176,122 if scene=='hands' else 137,reach=.66*extend)
        person(d,185,164 if scene=='hands' else 176,118 if scene=='hands' else 133,True,-1,.66*extend)
        if scene=='hands' and t<18.35:
            gray=im.convert('L').convert('RGB')
            im=Image.blend(gray,im,.19+.81*ramp(t,17.9,18.35)); d=ImageDraw.Draw(im)
        pulse_time = 19.65 if scene=='hands' else 27.29
        pulse = max(0, 1-abs(t-pulse_time)/.54)
        if pulse:
            glow = Image.new('RGBA',SIZE)
            heart(ImageDraw.Draw(glow),160,95 if scene=='hands' else 99,4*pulse,(255,119,156,200))
            halo = glow.filter(ImageFilter.GaussianBlur(3*S))
            im.paste(halo,(0,0),halo)
            im.paste(glow,(0,0),glow)
    else:
        sit=(scene!='bench' or t>=20.70) and (scene!='spring' or t<25.40)
        if sit: person(d,150,154,88,seated=True)
        else: person(d,146,162,113)
        if scene=='spring' and t>=25.04:
            u=ramp(t,25.04,26.06)
            person(d,332-97*u,163,113,True,-1,step=math.sin(u*math.pi*4.5))
        if scene=='snow': weather(d,t,True)
    if scene in ('heart','umbrella','window','rain','cinema','hands','bench','autumn','snow','evening'):
        mark(d,278,47)
    else: mark(d,38,124)
    return im


def draw_frame(n):
    t=n/FPS
    index=max(i for i,(start,_) in enumerate(SCENES) if t>=start)
    start,name=SCENES[index]
    frame=shot(name,t)
    if index+1<len(SCENES):
        switch,next_name=SCENES[index+1]
        duration=0 if next_name in ('credits','snow','evening') else .20
        if duration and t>switch-duration:
            frame=Image.blend(frame,shot(next_name,t),ramp(t,switch-duration,switch))
    if name=='heart' and t<.17: frame=Image.blend(Image.new('RGB',SIZE),frame,ramp(t,0,.17))
    if name=='together' and t>28:
        frame=Image.blend(frame,Image.new('RGB',SIZE),.78*ramp(t,28,28.77))
    for start,end,text in CAPTIONS:
        if start<=t<end:
            overlay=Image.new('RGBA',SIZE)
            d=ImageDraw.Draw(overlay)
            chunks=text.split('*')
            x=(SIZE[0]-sum(d.textlength(s,font=CAPTION_FONT) for s in chunks))/2
            fade=min(ramp(t,start,start+.08),1-ramp(t,end-.08,end))
            for i,s in enumerate(chunks):
                color=(220,73,99) if i%2 else (246,242,246)
                d.text((round(x),165*S),s,font=CAPTION_FONT,anchor='lm',fill=(*color,round(255*fade)),
                       stroke_width=1,stroke_fill=(32,23,35,round(190*fade)))
                x+=d.textlength(s,font=CAPTION_FONT)
            frame.paste(overlay,(0,0),overlay)
            break
    return frame


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',nargs='?',type=Path,default=Path('VID_20261127_073147_828.mp4'))
    parser.add_argument('-o','--output',type=Path,default=Path('drawn_animation.mp4'))
    args=parser.parse_args()
    if not args.source.is_file(): parser.error('Укажите путь к исходному MP4 со звуком.')
    if args.source.resolve()==args.output.resolve(): parser.error('Выберите отдельный выходной файл.')
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg=get_ffmpeg_exe()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
        target=Path(tmp)/'render.mp4'
        command=[ffmpeg,'-v','error','-f','rawvideo','-pixel_format','rgb24','-video_size',
                 f'{SIZE[0]}x{SIZE[1]}','-framerate',str(FPS),'-i','pipe:0','-i',str(args.source),
                 '-map','0:v:0','-map','1:a:0','-vf','scale=1280:720:flags=lanczos:out_color_matrix=bt709,setsar=1',
                 '-c:v','libx264','-preset','fast','-crf','16','-pix_fmt','yuv420p','-colorspace','bt709',
                 '-color_trc','bt709','-color_primaries','bt709','-c:a','copy','-movflags','+faststart',str(target)]
        with subprocess.Popen(command,stdin=subprocess.PIPE) as proc:
            try:
                for n in range(FRAMES):
                    proc.stdin.write(draw_frame(n).tobytes())
            finally:
                proc.stdin.close()
            if proc.wait(): raise SystemExit('FFmpeg не смог записать MP4.')
        target.replace(args.output)
    print(args.output.resolve())


if __name__=='__main__':
    main()
