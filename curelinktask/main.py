import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datetime import datetime


def load_data(file_path):
    with open('C:/newfile/queries.json', 'r') as f:
        return json.load(f)


def compare_meal_with_chart(latest_query, diet_chart, patient_profile):
    if not latest_query or not diet_chart or not patient_profile:
        return {
            'ideal_meal': 'No data available',
            'actual_meal': 'No data available',
            'missing_items': [],
            'extra_items': [],
            'non_compliant_items': []
        }

    current_time = datetime.now()
    start_date_str = diet_chart.get('start_date', '')
    if not start_date_str:
        return {
            'ideal_meal': 'No data available',
            'actual_meal': 'No data available',
            'missing_items': [],
            'extra_items': [],
            'non_compliant_items': []
        }

    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%dT%H:%M:%SZ')
    except ValueError:
        return {
            'ideal_meal': 'Invalid date format',
            'actual_meal': 'No data available',
            'missing_items': [],
            'extra_items': [],
            'non_compliant_items': []
        }

    day_number = (current_time - start_date).days + 1
    hour = current_time.hour
    if 4 <= hour < 10:
        meal_type = 'breakfast'
    elif 10 <= hour < 14:
        meal_type = 'lunch'
    elif 14 <= hour < 18:
        meal_type = 'snack'
    elif 18 <= hour < 23:
        meal_type = 'dinner'
    else:
        meal_type = 'late_night'

    ideal_meal = diet_chart.get('meals', {}).get(str(day_number), {}).get(meal_type,
                                                                          'No specific meal prescribed for this time')

    actual_meal = latest_query[0].get('message',
                                      'No meal information provided') if latest_query else "No meal information provided"

    missing_items = []
    extra_items = []

    ideal_items = set(ideal_meal.lower().split(', '))
    actual_items = set(actual_meal.lower().split(', '))

    for item in ideal_items:
        if item not in actual_items:
            missing_items.append(item)

    for item in actual_items:
        if item not in ideal_items:
            extra_items.append(item)

    health_conditions = patient_profile.get('health_conditions', [])
    non_compliant_items = []
    for item in actual_items:
        for condition in health_conditions:
            if condition.lower() in item:
                non_compliant_items.append(item)

    comparison_result = {
        'ideal_meal': ideal_meal,
        'actual_meal': actual_meal,
        'missing_items': missing_items,
        'extra_items': extra_items,
        'non_compliant_items': non_compliant_items
    }

    return comparison_result


def generate_response(context):
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    prompt = f"Patient profile: {context.get('patient_profile', '')}\n"
    prompt += f"Diet chart: {context.get('diet_chart', '')}\n"
    prompt += f"Latest query: {context.get('latest_query', '')}\n"
    prompt += f"Chat history: {context.get('chat_history', '')}\n"
    prompt += f"Meal compliance: {context.get('meal_compliance', '')}\n"
    prompt += "Based on the above information, provide a concise and actionable advice for the patient:\n"

    inputs = tokenizer.encode(prompt, return_tensors='pt', max_length=1024, truncation=True)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=1074,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    advice_start_index = len(prompt)
    advice = response[advice_start_index:].strip()

    return advice


def process_queries(data):
    output = []

    for query in data:
        try:
            profile_context = query.get('profile_context', {})
            chat_context = query.get('chat_context', {})

            patient_profile = profile_context.get('patient_profile', {})
            diet_chart = profile_context.get('diet_chart', {})
            latest_query = query.get('latest_query', [])
            chat_history = chat_context.get('chat_history', '')

            if not isinstance(patient_profile, dict):
                patient_profile = {}
            if not isinstance(diet_chart, dict):
                diet_chart = {}

            meal_compliance = compare_meal_with_chart(latest_query, diet_chart, patient_profile)

            context = {
                'patient_profile': patient_profile,
                'diet_chart': diet_chart,
                'latest_query': latest_query,
                'chat_history': chat_history,
                'meal_compliance': meal_compliance
            }

            generated_response = generate_response(context)

            output.append({
                'ticket_id': chat_context.get('ticket_id', ''),
                'latest_query': latest_query,
                'generated_response': generated_response,
                'ideal_response': query.get('ideal_response', '')
            })

        except KeyError as e:
            print(f"KeyError: {e} in query: {query}")
        except IndexError as e:
            print(f"IndexError: {e} in query: {query}")
        except Exception as e:
            print(f"Unexpected error: {e}")

    return output


def save_output(output, file_path):
    with open('C:/newfile/output.json', 'w') as f:
        json.dump(output, f, indent=4)


if __name__ == "__main__":
    try:
        data = load_data(r'C:\newfile\queries.json')
        output = process_queries(data)
        save_output(output, r'C:\newfile\output.json')
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
