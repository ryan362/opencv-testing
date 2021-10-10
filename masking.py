import cv2 as cv
import numpy as np

img = cv.imread('Resources/Photos/cats 2.jpg')
# cv.imshow('Cats', img)

blank = np.zeros(img.shape[:2], dtype='uint8')
# cv.imshow('Blank', blank)

# mask = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)
# mask1 = cv.circle(blank, (img.shape[1]//2 + 45, img.shape[0]//2), 100, 255, -1)
# recmask = cv.rectangle(blank, (img.shape[1]//2, img.shape[0]//2), (img.shape[1]//2 + 100, img.shape[0]//2 + 100), 255, -1)
# cv.imshow('Mask', mask)
# cv.imshow('Mask1', mask1)

circle = cv.circle(blank.copy(), (img.shape[1]//2 + 45,img.shape[0]//2), 100, 255, -1)
rectangle = cv.rectangle(blank.copy(), (30,30), (370,370), 255, -1)

weird_shape = cv.bitwise_and(circle, rectangle)
# cv.imshow('WeirdShape', weird_shape)

weirdMask = cv.bitwise_and(img, img, mask=weird_shape)
cv.imshow('WeirdMask', weirdMask)

# masked = cv.bitwise_and(img, img, mask=mask)
# cv.imshow('Masked', masked) 
# masked1 = cv.bitwise_and(img, img, mask=mask1)
# # cv.imshow('Masked1', masked1)
# recmask = cv.bitwise_and(img, img, mask=recmask)
# cv.imshow('RecMask', recmask)

cv.waitKey(0)