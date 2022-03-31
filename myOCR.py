import logging
from difflib import get_close_matches

import click
import cv2
import nltk
import numpy as np
import pytesseract
from enchant.checker import SpellChecker
from enchant.tokenize import EmailFilter, URLFilter, WikiWordFilter
from nltk.tag.stanford import StanfordNERTagger
from nltk.tokenize import word_tokenize
from pdf2image import convert_from_path

nltk.download('punkt')
st = StanfordNERTagger(
    '/content/stanford-ner-2020-11-17/classifiers/english.all.3class.distsim.crf.ser.gz',
    '/content/stanford-ner-2020-11-17/stanford-ner.jar',
    encoding='utf-8')


# Calculate skew angle of an image
def getSkewAngle(img):
    blur = cv2.GaussianBlur(img, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=5)

    # Find all contours
    contours, _ = cv2.findContours(dilate, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)

    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle


# Rotate the image around its center
def rotateImage(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(img,
                         M, (w, h),
                         flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REPLICATE)
    return img


# Deskew image
def deskew(img):
    angle = getSkewAngle(img)
    return rotateImage(img, -1.0 * angle)


def removeUnderscores(img):
    thresh = cv2.threshold(img, 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    detected_lines = cv2.morphologyEx(thresh,
                                      cv2.MORPH_OPEN,
                                      horizontal_kernel,
                                      iterations=2)
    countours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    for c in countours:
        cv2.drawContours(img, [c], -1, (255, 255, 255), 2)
    return img


def removeNoise(img):
    blur = cv2.GaussianBlur(img, (1, 1), 0)
    thresh = cv2.threshold(blur, 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    adp_thresh = cv2.adaptiveThreshold(blur, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

    # combine the result of two thresholds, hope can get better recognition
    return cv2.bitwise_or(thresh, adp_thresh)


def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    deskewed = deskew(gray)
    denoised = removeNoise(deskewed)
    remove_ = removeUnderscores(denoised)
    return remove_


def runOCR(img):
    custom_config = r'-l eng --oem 3 --psm 6'
    txt = pytesseract.image_to_string(img, config=custom_config)
    return txt


def getNamedEntities(txt):
    named_entities = [
        entity[0] for entity in st.tag(word_tokenize(txt))
        if entity[1] == 'PERSON' or entity[1] == 'ORGANIZATION'
    ]
    named_entities = list(set(named_entities))  # remove duplicates
    return named_entities


def postprocess(txt):
    # use filters to avoid over-correcting
    checker = SpellChecker("en_US",
                           filters=[EmailFilter, URLFilter, WikiWordFilter])
    checker.set_text(txt)

    named_entities = getNamedEntities(txt)

    # correct error words
    for err in checker:
        # to avoid over-correcting, keep these cases unchanged
        if err.word.isupper():  # maybe acronym
            continue
        if err.word.endswith("'s"):  # maybe possessive
            continue
        if err.word in named_entities:  # maybe a name
            continue

        # get suggested correct words
        suggested = checker.suggest(err.word)
        # to get the most similiar word, hope can improve the accuracy
        matches = get_close_matches(err.word, suggested)

        if matches:  # select the most similiar word
            correction = matches[0]
        elif suggested:  # no match, select the first suggested word
            correction = suggested[0]
        else:  # no match & suggested, keep the word unchanged
            correction = err.word
        err.replace(correction)

    # return corrected text
    return checker.get_text()


@click.command()
@click.option('--input', '-i', help='Input file')
@click.option('--output', '-o', default='output.txt', help='Output file')
@click.option('--verbose', '-v', is_flag=True, help='Print verbose')
def myOCR(input, output, verbose):
    """A command‐line tool to do OCR and extract text from a given scanned document image."""

    LOG_FORMAT = "%(asctime)s %(name)s -> %(levelname)s: %(message)s"
    logging.basicConfig(format=LOG_FORMAT)
    logger = logging.getLogger(__name__)
    if verbose:
        logger.setLevel(logging.INFO)

    assert input is not None, 'Input file is required!'

    in_ext = input.split('.')[-1].lower()
    if in_ext == 'pdf':  # convert to images
        images = convert_from_path(input, dpi=300)
        images = [np.array(img) for img in images]
    elif in_ext == 'jpg' or in_ext == 'png':
        images = [cv2.imread(input)]
    else:
        raise TypeError('Unknown file type!')
    logger.info(f'Read in {in_ext.upper()} file')

    logger.info('Start processing...')
    preprocessed_img = [preprocess(img) for img in images]
    txt = '\n'.join([runOCR(img) for img in preprocessed_img])
    result = postprocess(txt)

    with open(output, 'w') as fout:
        fout.write(result)
    logger.info('Finished!')


if __name__ == '__main__':
    myOCR()
