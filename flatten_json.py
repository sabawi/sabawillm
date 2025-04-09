import json

def flatten_json(input_file, output_file):
    try:
        # Read the input JSON file
        with open(input_file, 'r') as file:
            input_data = json.load(file)

        # Flatten the nested JSON structure into a single list
        flattened_list = [item for sublist in input_data for item in sublist]

        # Save the flattened list to the output JSON file
        with open(output_file, 'w') as file:
            json.dump(flattened_list, file, indent=4)

        print(f"Flattened data has been successfully saved to '{output_file}'.")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found. Please check the file path.")
    except json.JSONDecodeError:
        print(f"Error: The file '{input_file}' does not contain valid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Prompt the user for input and output file names
input_file = input("Enter the name of the input JSON file (e.g., input.json): ").strip()
output_file = input("Enter the name of the output JSON file (e.g., output.json): ").strip()

# Call the function to flatten the JSON
flatten_json(input_file, output_file)