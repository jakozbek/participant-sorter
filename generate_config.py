#!/usr/bin/env python3

import pandas as pd
import json
import sys
from collections import defaultdict


def analyze_csv_structure(csv_file_path):
    """
    Analyze a CSV file to identify class columns and time slots.

    Args:
        csv_file_path (str): Path to the CSV file to analyze

    Returns:
        dict: Configuration dictionary with class and time slot information
    """
    # Read CSV file
    df = pd.read_csv(csv_file_path)

    print(f"Analyzing CSV file: {csv_file_path}")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    # Identify class columns (same logic as main script)
    class_columns = [
        col
        for col in df.columns
        if "]" in col and "[" in col and "(" in col and ")" in col
    ]

    # Extract time slots
    time_slots = set(col.split("[")[0].strip() for col in class_columns)

    # Group classes by time slot
    time_slot_classes = defaultdict(list)
    for col in class_columns:
        slot = col.split("[")[0].strip()
        class_info = col.split("[")[1].strip("]")
        # Extract class name and teacher
        class_name = None
        teacher = None
        if "(" in class_info and ")" in class_info:
            # Split at the last opening parenthesis to separate class name from teacher
            paren_start = class_info.rfind("(")
            class_name = class_info[:paren_start].strip()
            teacher = class_info[paren_start + 1 :].rstrip(")").strip()
        else:
            # If no parentheses, the whole thing is the class name
            class_name = class_info.strip()

        time_slot_classes[slot].append(
            {
                "full_name": col,
                "class_info": class_info,
                "class_name": class_name,
                "teacher": teacher,
                "time_slot": slot,
            }
        )

    # Check for participant name column
    name_column = None
    name_candidates = [
        "What is your full name?",
        "Email Address",
        "Name",
        "Full Name",
        "Participant Name",
    ]

    for candidate in name_candidates:
        if candidate in df.columns:
            name_column = candidate
            break

    if not name_column:
        # Look for any column with 'name' in it
        name_cols = [col for col in df.columns if "name" in col.lower()]
        if name_cols:
            name_column = name_cols[0]

    # Check for timestamp column
    timestamp_column = None
    if "Timestamp" in df.columns:
        timestamp_column = "Timestamp"

    # Analyze choice values in class columns
    choice_values = set()
    for col in class_columns:
        unique_values = df[col].dropna().unique()
        choice_values.update(unique_values)

    # Filter out empty strings and common non-choice values
    choice_values = [val for val in choice_values if val and str(val).strip()]

    # Calculate popularity scores for each class option
    popularity_scores = defaultdict(
        lambda: {"total_points": 0, "choice_breakdown": defaultdict(int)}
    )

    # Point values for choices (1st choice = 5 points, 2nd = 4, etc.)
    choice_points = {
        "1st choice": 5,
        "2nd choice": 4,
        "3rd choice": 3,
        "4th choice": 2,
        "5th choice": 1,
    }

    # Analyze each participant's choices
    for _, row in df.iterrows():
        for col in class_columns:
            choice = row[col]
            if pd.notna(choice) and choice in choice_points:
                points = choice_points[choice]
                popularity_scores[col]["total_points"] += points
                popularity_scores[col]["choice_breakdown"][choice] += 1

    # Extract unique class names and track their time slots
    unique_classes = set()
    unique_teachers = set()
    class_time_slots = defaultdict(list)
    class_instructors = defaultdict(set)

    for col in class_columns:
        slot = col.split("[")[0].strip()
        class_info = col.split("[")[1].strip("]")
        if "(" in class_info and ")" in class_info:
            paren_start = class_info.rfind("(")
            class_name = class_info[:paren_start].strip()
            teacher = class_info[paren_start + 1 :].rstrip(")").strip()
            unique_classes.add(class_name)
            if teacher:
                unique_teachers.add(teacher)
                class_instructors[class_name].add(teacher)
        else:
            class_name = class_info.strip()
            unique_classes.add(class_name)

        # Track which time slots this class is offered in
        class_time_slots[class_name].append(slot)

    # Create class configuration
    class_config = {}
    for class_name in sorted(unique_classes):
        class_config[class_name] = {
            "capacity": 20,  # To be filled in by user
            "required_classes": [],  # List of class names that must be taken first
            "time_slots": sorted(
                list(set(class_time_slots[class_name]))
            ),  # Remove duplicates and sort
            "instructors": sorted(
                list(class_instructors[class_name])
            ),  # All instructors for this class
            "description": f"Configuration for {class_name}",
        }

    # Create popularity analysis
    popularity_analysis = {}
    for col in class_columns:
        slot = col.split("[")[0].strip()
        class_info = col.split("[")[1].strip("]")

        # Extract class name (same logic as above)
        if "(" in class_info and ")" in class_info:
            paren_start = class_info.rfind("(")
            class_name = class_info[:paren_start].strip()
            teacher = class_info[paren_start + 1 :].rstrip(")").strip()
        else:
            class_name = class_info.strip()
            teacher = None

        popularity_analysis[col] = {
            "class_name": class_name,
            "teacher": teacher,
            "time_slot": slot,
            "total_points": popularity_scores[col]["total_points"],
            "choice_breakdown": dict(popularity_scores[col]["choice_breakdown"]),
            "total_responses": sum(popularity_scores[col]["choice_breakdown"].values()),
        }

    config = {
        "csv_info": {
            "file_path": csv_file_path,
            "total_participants": len(df),
            "participant_name_column": name_column,
            "timestamp_column": timestamp_column,
        },
        "classes": class_config,
        "unique_class_names": sorted(list(unique_classes)),
        "total_unique_classes": len(unique_classes),
        "popularity_analysis": popularity_analysis,
    }

    return config


def generate_config_file(csv_file_path, output_config_path=None):
    """
    Generate a configuration file based on CSV analysis.

    Args:
        csv_file_path (str): Path to the CSV file to analyze
        output_config_path (str, optional): Path for output config file
    """
    try:
        config = analyze_csv_structure(csv_file_path)

        # Generate output filename if not provided
        if not output_config_path:
            base_name = csv_file_path.replace(".csv", "")
            output_config_path = f"{base_name}_config.json"

        # Save configuration to JSON file
        with open(output_config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Print summary
        print(f"\nConfiguration Analysis Complete!")
        print("=" * 40)
        print(
            f"Participant name column: {config['csv_info']['participant_name_column']}"
        )
        print(f"Timestamp column: {config['csv_info']['timestamp_column']}")
        print(f"Total participants: {config['csv_info']['total_participants']}")
        print(f"Total unique classes: {config['total_unique_classes']}")

        print(f"\nUnique Classes Found:")
        for class_name in config["unique_class_names"]:
            print(f"  {class_name}")

        # Print popularity analysis
        print(f"\nClass Popularity Analysis (sorted by total points):")
        print("=" * 60)

        # Sort by total points descending
        sorted_popularity = sorted(
            config["popularity_analysis"].items(),
            key=lambda x: x[1]["total_points"],
            reverse=True,
        )

        for col_name, data in sorted_popularity:
            if data["total_points"] > 0:  # Only show classes with responses
                print(f"\n{data['class_name']} - {data['time_slot']}")
                if data["teacher"]:
                    print(f"  Teacher: {data['teacher']}")
                print(f"  Total Points: {data['total_points']}")
                print(f"  Total Responses: {data['total_responses']}")
                print(f"  Choice Breakdown:")
                for choice, count in sorted(data["choice_breakdown"].items()):
                    print(f"    {choice}: {count} participants")

        print(f"\nNext Steps:")
        print(f"1. Edit {output_config_path}")
        print(f"2. Set capacity values for each class in 'classes'")
        print(f"3. Use popularity data to inform capacity decisions")
        print(f"4. Use this config file with your assignment script")

        print(f"\nConfiguration saved to: {output_config_path}")

        return config

    except Exception as e:
        print(f"Error analyzing CSV file: {e}")
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_config.py <csv_file_path> [output_config_path]")
        sys.exit(1)

    csv_file_path = sys.argv[1]
    output_config_path = sys.argv[2] if len(sys.argv) > 2 else None

    generate_config_file(csv_file_path, output_config_path)
