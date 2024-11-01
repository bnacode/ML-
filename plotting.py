import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from common import train_and_test_slit


def plot_students_score(Y):
    # passing_threshold = 10
    passing_threshold = 10

    # Determine passing and failing status
    passed = Y >= passing_threshold
    failed = ~passed

    # Count the number of students who passed and failed
    num_passed = np.sum(passed)
    num_failed = np.sum(failed)

    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Plotting the distribution of passing and failing students
    labels_passed_failed = ['Passed', 'Failed']
    values_passed_failed = [num_passed, num_failed]
    ax1.bar(labels_passed_failed, values_passed_failed, color=['green', 'red'])
    ax1.set_xlabel('Status')
    ax1.set_ylabel('Number of Students')
    ax1.set_title('Number of Students Passed and Failed')

    # Plotting the distribution of student scores
    score_classes = {
        'A': range(16, 21),
        'B': range(14, 16),
        'C': range(12, 14),
        'D': range(10, 12),
        'F': range(0, 10),
    }

    # Categorize scores
    score_categories = {grade: category for category, scores in score_classes.items() for grade in scores}
    categories = np.array([score_categories[grade] for grade in Y])

    # Count the number of students in each category
    category_counts = {category: np.sum(categories == category) for category in score_classes.keys()}

    # Plotting the distribution of student scores
    labels_score_categories = list(score_classes.keys())
    values_score_categories = [category_counts[category] for category in labels_score_categories]
    ax2.bar(labels_score_categories, values_score_categories, color=['green', 'blue', 'orange', 'yellow', 'red'])
    ax2.set_xlabel('Score Categories')
    ax2.set_ylabel('Number of Students')
    ax2.set_title('Distribution of Student Scores')

    # Plotting the distribution of individual student scores
    ax3.hist(Y, bins=np.arange(0, 22), edgecolor='black', color='skyblue')
    ax3.set_xlabel('Student Scores')
    ax3.set_ylabel('Number of Students')
    ax3.set_title('Distribution of Individual Student Scores')

    plt.show()
