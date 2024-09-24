import pandas as pd
import numpy as np

# Generate tabular data of 1000 samples. Each row is a person who was either hired or not hired by Google for an internship. The data has 2 columns: average grade, hired (1 if hired, 0 if not hired).
n = 20000

rows = []
for _ in range(n):
    avg_grade = 60 + np.random.normal(0, 15)
    avg_grade = np.maximum(avg_grade, 0)
    avg_grade = np.minimum(avg_grade, 100)

    # The probability of being hired is a sigmoid function of the average grade
    prob_hired = 1 / (1 + np.exp(-0.04 * avg_grade + 8))
    hired = np.random.rand() < prob_hired

    rows.append({"Average grade": avg_grade, "Hired": hired})

df = pd.DataFrame(rows)

# Save the dataframe to a csv file
df.to_csv(
    "../IntroToML/data/googleinternship_simple.csv",
    index=False,
)

# Now new data with 20 columns:
# - Average grade
# - Number of projects
# - Number of previous internships
# - Number of hackathons
# - Number of published research papers
# - Number of programming languages known
# - Did sports at university (1 if yes, 0 if no)
# - Has previous google internship (1 if yes, 0 if no)
# - University ranking
# - Number of programming competitions won
# - Number of programming competitions participated in
# - Hired (1 if hired, 0 if not hired)

rows = []

n = 50000


def generate_norm_int_from_strength(strength_of_candidate, upper_mean, std):
    norm = np.maximum(0, np.random.normal(strength_of_candidate * upper_mean, std))
    return norm.astype(int)


for _ in range(n):
    strength_of_candidate = np.random.normal(0.2, 0.6)
    strength_of_candidate = np.maximum(strength_of_candidate, 0)
    avg_grade = 60 + np.random.normal(strength_of_candidate * 10, 15)
    avg_grade = np.maximum(avg_grade, 0)
    avg_grade = np.minimum(avg_grade, 100)

    number_of_projects = generate_norm_int_from_strength(strength_of_candidate, 5, 2)
    number_of_previous_internships = generate_norm_int_from_strength(
        strength_of_candidate, 2, 1
    )
    number_of_hackathons = generate_norm_int_from_strength(strength_of_candidate, 2, 1)
    number_of_published_research_papers = generate_norm_int_from_strength(
        strength_of_candidate, 0.5, 1
    )
    number_of_programming_languages_known = generate_norm_int_from_strength(
        strength_of_candidate, 5, 2
    )
    did_sports_at_university = np.random.rand() > 0.6
    has_previous_google_internship = np.random.rand() > 0.999
    university_ranking = generate_norm_int_from_strength(strength_of_candidate, 100, 50)
    number_of_programming_competitions_won = generate_norm_int_from_strength(
        strength_of_candidate, 0.5, 0.1
    )
    number_of_programming_competitions_participated_in = (
        generate_norm_int_from_strength(strength_of_candidate, 2, 1)
    )

    if has_previous_google_internship:
        hired = np.random.rand() > 0.05
    elif (
        number_of_programming_competitions_won > 0
        and number_of_previous_internships > 0
        and avg_grade > 80
    ):
        hired = np.random.rand() > 0.1
    elif (
        number_of_programming_competitions_participated_in > 3
        and number_of_projects > 3
        and number_of_previous_internships > 1
        and avg_grade > 70
    ):
        hired = np.random.rand() > 0.2
    else:
        hired = np.random.rand() > 0.999

    rows.append(
        {
            "Average grade": avg_grade,
            "Number of projects": number_of_projects,
            "Number of previous internships": number_of_previous_internships,
            "Number of hackathons": number_of_hackathons,
            "Number of published research papers": number_of_published_research_papers,
            "Number of programming languages known": number_of_programming_languages_known,
            "Did sports at university": did_sports_at_university,
            "Has previous google internship": has_previous_google_internship,
            "University ranking": university_ranking,
            "Number of programming competitions won": number_of_programming_competitions_won,
            "Number of programming competitions participated in": number_of_programming_competitions_participated_in,
            "Hired": hired,
        }
    )

df = pd.DataFrame(rows)

# Save the dataframe to a csv file
df.to_csv(
    "./IntroToML/data/googleinternship_big.csv",
    index=False,
)
