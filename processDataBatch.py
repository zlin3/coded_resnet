import random

L5 = b'\x05'
L8 = b'\x08'

LABEL_BYTES = 1
PIXEL_BYTES = 3072

def process(fileName):
    data = []
    with open(fileName, 'rb') as f:
        data = f.read()
    labels = data[0::(PIXEL_BYTES+LABEL_BYTES)]
    pixels = []
    for i in range(LABEL_BYTES, len(data), PIXEL_BYTES+LABEL_BYTES):
        pixels.append(data[i:i+PIXEL_BYTES])
    pixels5 = []
    pixels8 = []
    pixelsElse = []
    labelsElse = []
    for i, label in enumerate(labels):
        if label == L5:
           pixels5.append(pixels[i])
        elif label == L8:
           pixels8.append(pixels[i])
        else:
           pixelsElse.append(pixels[i])
           labelsElse.append(label)
    
    return pixels5, pixels8, pixelsElse, labelsElse, len(labels)

def generate(totalLen, fileName, pixels5, pixels8, pixelsElse, labelsElse):
    data = []
    labels = [L5]*len(pixels5) + [L8]*len(pixels8) 
    pixels = pixels5 + pixels8
    remaining = totalLen / 2 - len(labels)
    print remaining
    for i in range(remaining):
        p = random.random()
        if p <= 0.5:
            labels.append(L5)
            pixels.append(random.choice(pixels5))
        else:
            labels.append(L8)
            pixels.append(random.choice(pixels8))
    print len(labels), len(pixels)
    randomIdx = random.sample(range(0, len(labelsElse)), totalLen / 2)
    for idx in randomIdx:
        labels.append(labelsElse[idx]) 
        pixels.append(pixelsElse[idx])
    indices = range(totalLen)
    #random.shuffle(indices)
    with open(fileName, 'wb') as f:
        for idx in indices:
            data.append(labels[idx])
            f.write(labels[idx])
            pixel = pixels[idx]
            for b in pixel:
                data.append(b)
                f.write(b)
    print len(labels), len(pixels)


def main():
    pixels5, pixels8, pixelsElse, labelsElse, total = process('data_batch_1.bin')
    generate(total, 'test.bin', pixels5, pixels8, pixelsElse, labelsElse)
    pixels5, pixels8, pixelsElse, labelsElse, total = process('test.bin')
    return pixels5, pixels8, pixelsElse, labelsElse, total 
     
