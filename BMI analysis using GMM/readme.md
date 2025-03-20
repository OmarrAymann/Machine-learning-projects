# Gaussian Mixture Models (GMM) for Weight-Height Dataset and BMI Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.0%2B-green)
![Pandas](https://img.shields.io/badge/Pandas-1.0%2B-red)

## Introduction

This project explores the use of **Gaussian Mixture Models (GMM)**, a probabilistic clustering algorithm, to analyze a dataset containing weight and height measurements. The goal is to group individuals into clusters based on their physical characteristics and derive insights from their **Body Mass Index (BMI)**, a widely used metric to assess health based on weight and height.

# BMI Calculator

## Formula

The Body Mass Index (BMI) is calculated using the formula:

![BMI Formula](https://latex.codecogs.com/png.latex?BMI%20%3D%20%5Cfrac%7Bweight%20(kg)%7D%7Bheight%20(m)%5E2%7D)


GMM is particularly useful for this task because it can identify overlapping clusters and assign probabilistic memberships to data points. By applying GMM, we aim to uncover natural groupings in the dataset, such as underweight, normal weight, overweight, and obese categories, based on BMI values.

This project demonstrates how unsupervised learning techniques like GMM can be applied to real-world health data, providing valuable insights into population health trends and aiding in personalized health recommendations.

## Dataset

The dataset used in this project contains the following columns:
- **Weight (kg):** The weight of individuals in kilograms.
- **Height (m):** The height of individuals in meters.

The dataset is preprocessed to calculate BMI for each individual, which is then used as a feature for clustering.

## Usage

To calculate BMI, input weight in kilograms and height in meters. The result helps determine if a person is underweight, normal weight, overweight, or obese.

## Categories

| BMI Range       | Category         |
|----------------|----------------|
| Below 18.5     | Underweight     |
| 18.5 - 24.9    | Normal weight   |
| 25 - 29.9      | Overweight      |
| 30 and above   | Obese           |

## Example

If a person weighs **70 kg** and is **1.75 m** tall:

$$
BMI = \frac{70}{(1.75)^2}  = 22.86
$$

This falls into the **Normal weight** category.

---

