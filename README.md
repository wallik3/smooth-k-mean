# Smooth-k-mean: Handling Imbalanced Data with Equilibrium K-means (EKM)

## Overview

Smooth-k-mean is a Python implementation of the Equilibrium K-means (EKM) algorithm proposed by Yudong He to address the issue of learning bias towards large clusters in centroid-based clustering algorithms, especially in the context of imbalanced data. This README provides an overview of the algorithm, its objectives, and its implementation details.

## Motivation

Imbalanced data, where the true underlying groups of data have different sizes, is a common challenge in various domains such as medical diagnosis, fraud detection, and anomaly detection. Traditional centroid-based clustering algorithms like hard K-means (HKM) and fuzzy K-means (FKM) tend to exhibit learning bias towards large clusters, compromising their performance when faced with imbalanced data. Smooth-k-mean aims to mitigate this bias by introducing a novel centroid repulsion mechanism based on the Boltzmann operator.

## Algorithm Overview

Smooth-k-mean, or Equilibrium K-means (EKM), introduces a new clustering objective function that incorporates a centroid repulsion mechanism. In this mechanism, data points surrounding centroids repel other centroids, with larger clusters exerting stronger repulsion forces. This effectively mitigates the issue of large cluster learning bias. The algorithm alternates between two steps, making it resource-saving with the same time and space complexity as FKM. It is also scalable to large datasets via batch learning.

## Features

- Implementation of the Equilibrium K-means (EKM) algorithm
- Designed to handle imbalanced data by mitigating large cluster learning bias
- Simple, alternating between two steps
- Resource-saving with the same time and space complexity as FKM
- Scalable to large datasets via batch learning

## Evaluation

Smooth-k-mean has been substantially evaluated on synthetic and real-world datasets. The results demonstrate its competitive performance on balanced data and its significant improvement over benchmark algorithms on imbalanced data. Deep clustering experiments further highlight Smooth-k-mean as a better alternative to HKM and FKM on imbalanced data, as it yields more discriminative representations.

## Installation

To use Smooth-k-mean, simply clone this repository and install the required dependencies:

```bash
git clone <repository-url>
cd smooth-k-mean
pip install -r requirements.txt
