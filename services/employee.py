from dependencies import MongoDB

client = MongoDB()

sentiment_values = {
    "Positive": 1,
    "Negative": -1,
    "Neutral": 0
}

sentiment_fields = {
    "Positive": "num_positive_conversations",
    "Negative": "num_negative_conversations",
    "Neutral": "num_neutral_conversations"
}


def _get_sentiment(score):
    if score < -0.33:
        return "Negative"
    elif score > 0.33:
        return "Positive"
    else:
        return "Neutral"

def Update_Employee_Stats(email,score,sentiment):
    employee = client.db.employees.find_one({'employee_email':email})
    num_conversations = employee["num_conversations"] + 1
    cumulative_score = employee["cumulative_score"] + score*sentiment_values[sentiment]
    updatable_field = sentiment_fields[sentiment]
    updatable_field_value = employee[updatable_field] + 1
    score = cumulative_score/num_conversations
    result = client.db.employees.update_one(
        {"employee_email": email},
        {
            "$set": {
                "score": score,
                "num_conversations": num_conversations,
                "cumulative_score":cumulative_score,
                "average_sentiment": _get_sentiment(cumulative_score),
                updatable_field : updatable_field_value
            }
        }
    )

    if result.modified_count == 1:
        print("modified employee stats successfully")
