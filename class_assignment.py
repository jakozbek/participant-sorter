#!/usr/bin/env python3

import pandas as pd
from collections import defaultdict


def assign_participants(df, class_capacity):
    """
    Assign participants to classes based on their preferences and class capacity.

    Args:
        df (pandas.DataFrame): DataFrame with participant preferences
        class_capacity (int): Maximum number of participants per class

    Returns:
        dict: Dictionary with class names as keys and lists of participant names as values
    """
    # Extract class columns and time slots
    class_columns = [
        col
        for col in df.columns
        if "]" in col and "[" in col and "(" in col and ")" in col
    ]
    print(f"Identified class columns: {class_columns}")

    time_slots = set(col.split("[")[0].strip() for col in class_columns)
    print(f"Identified time slots: {time_slots}")

    # Initialize assignments dictionary
    assignments = {col: [] for col in class_columns}

    # Track which time slots each user is assigned to
    user_assignments = defaultdict(set)

    # Track points for each user (lower points = higher priority)
    # 3 points for 1st choice, 2 for 2nd, 1 for 3rd
    user_points = defaultdict(int)

    # Create a dictionary to store timestamps for tiebreaking
    # Assuming there's a "Timestamp" column in the dataframe
    timestamps = {}
    if "Timestamp" in df.columns:
        for _, row in df.iterrows():
            timestamps[row["What is your full name?"]] = row["Timestamp"]

    # Process preferences in order: 1st choice, 2nd choice, 3rd choice
    for choice_idx, choice in enumerate(["1st choice", "2nd choice", "3rd choice"]):
        # Points awarded for this choice level (3, 2, or 1)
        choice_points = 3 - choice_idx

        # For each time slot, process all classes
        for slot in sorted(time_slots):
            slot_columns = [col for col in class_columns if slot in col]

            # First, identify all users who want each class as their current choice level
            class_seekers = defaultdict(list)
            for col in slot_columns:
                for _, row in df.iterrows():
                    full_name = row["What is your full name?"]
                    # Skip if user already assigned to this time slot
                    if slot in user_assignments[full_name]:
                        continue
                    # Check if this is their choice for this class
                    if row[col] == choice:
                        class_seekers[col].append(full_name)

            # Then assign users, prioritizing those with fewer points (less satisfied so far)
            for col in slot_columns:
                # Define a key function for sorting users
                def sort_key(full_name):
                    # Primary sort by points (lower is better)
                    # Secondary sort by timestamp if available (earlier is better)
                    if full_name in timestamps:
                        return (user_points[full_name], timestamps[full_name])
                    return (user_points[full_name], "")

                # Sort users by points and timestamp
                sorted_users = sorted(class_seekers[col], key=sort_key)

                # Assign users until capacity is reached
                for full_name in sorted_users:
                    if (
                        len(assignments[col]) < class_capacity
                        and slot not in user_assignments[full_name]
                    ):
                        assignments[col].append(full_name)
                        user_assignments[full_name].add(slot)

                        # Award points based on which choice they got
                        user_points[full_name] += choice_points

    # Handle any users not yet assigned by finding available spots
    for _, row in df.iterrows():
        full_name = row["What is your full name?"]
        for slot in sorted(time_slots):
            # Skip if user already has a class in this slot
            if slot in user_assignments[full_name]:
                continue

            # Find classes with availability in this slot
            available_classes = [
                col
                for col in class_columns
                if slot in col and len(assignments[col]) < class_capacity
            ]

            # Prioritize classes with fewer participants
            available_classes.sort(key=lambda col: len(assignments[col]))

            # Assign to first available class
            if available_classes:
                assignments[available_classes[0]].append(full_name)
                user_assignments[full_name].add(slot)
                # No points for fallback assignments

    # Print user points summary at the end
    print("\nParticipant Priority Scores:")
    print("============================")
    # Sort by name alphabetically for easier reading
    for full_name in sorted(user_points.keys()):
        print(f"{full_name}: {user_points[full_name]} points")

    return assignments


def time_slot_sort_key(slot):
    """Sort time slots chronologically"""
    # Extract time info from slots like "10 AM-12 PM", "1-3 PM", or "Saturday 9:00-10:15am"
    if "-" in slot:
        start_time = slot.split("-")[0].strip()

        print(f"Start time: {start_time}")

        # Check for day of week prefix (like "Saturday")
        day_prefix = 0
        days = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        for i, day in enumerate(days):
            if day in start_time:
                start_time = start_time.replace(day, "").strip()
                day_prefix = (
                    i + 1
                ) * 100  # Add day multiplier (100 for Monday, 200 for Tuesday, etc.)
                break

        # Handle hour formats
        if ":" in start_time:
            # Extract only digits before the colon
            hour_str = "".join(c for c in start_time.split(":")[0] if c.isdigit())
            hour = int(hour_str) if hour_str else 0
        else:
            # Extract only digits
            hour_str = "".join(c for c in start_time if c.isdigit())
            hour = int(hour_str) if hour_str else 0

        # Adjust for AM/PM
        # if "pm" in start_time.lower() and hour < 12:
        #     hour += 12
        # if "am" in start_time.lower() and hour == 12:
        #     hour = 0

        if "pm" in start_time.lower() and hour < 12:
            hour += 12  # 1pm becomes 13, etc.
        elif "am" in start_time.lower():
            if hour == 12:
                hour = 0  # 12am becomes 0

        print(day_prefix + hour)

        return day_prefix + hour
    return 0  # Default for any other format


def process_csv_file(csv_file_path, class_capacity, output_csv=None):
    """
    Process a CSV file of participant preferences and assign them to classes.

    Args:
        csv_file_path (str): Path to the CSV file with preferences
        class_capacity (int): Maximum number of participants per class
        output_csv (str, optional): Path to output CSV file for assignments

    Returns:
        dict: Dictionary with class names as keys and lists of participant names as values
    """
    # Read CSV file
    df = pd.read_csv(csv_file_path)

    # Display available columns to help with debugging
    print("Available columns in CSV:", df.columns.tolist())

    # Check if 'Email Address' exists, otherwise look for similar column
    name_column = "What is your full name?"
    if name_column not in df.columns:
        # Try to find alternative full_name column
        name_candidates = [col for col in df.columns if "full_name" in col.lower()]
        if name_candidates:
            name_column = name_candidates[0]
            print(f"Using '{name_column}' as full_name column")
            # Rename the column for consistency with the rest of the code
            df = df.rename(columns={name_column: "What is your full name?"})
        else:
            print("Error: No full_name column found in CSV file.")
            print("Please ensure your CSV has a column named 'Email Address'")
            return {}

    # Run assignment algorithm
    assignments = assign_participants(df, class_capacity)

    # Print results
    print("\nClass Assignments:")
    print("=================")

    # Group by time slot for better readability
    time_slots = set(col.split("[")[0].strip() for col in assignments.keys())

    for slot in sorted(time_slots, key=time_slot_sort_key):
        print(f"\n{slot}:")
        slot_classes = [col for col in assignments.keys() if slot in col]

        for class_name in sorted(slot_classes):
            class_info = class_name.split("[")[1].strip("]")
            print(
                f"  Class {class_info} ({len(assignments[class_name])}/{class_capacity}):"
            )
            for full_name in assignments[class_name]:
                print(f"    - {full_name}")

    # Check if any participant wasn't assigned to all time slots
    participants = df["What is your full name?"].tolist()
    participant_assignments = defaultdict(list)

    for class_name, names in assignments.items():
        time_slot = class_name.split("[")[0].strip()
        for full_name in names:
            participant_assignments[full_name].append(time_slot)

    print("\nParticipant Summary:")
    print("===================")
    for full_name in participants:
        assigned_slots = participant_assignments[full_name]
        missing_slots = [slot for slot in time_slots if slot not in assigned_slots]

        if missing_slots:
            print(f"{full_name}: Missing assignments for {', '.join(missing_slots)}")

    # Create output CSV if requested
    if output_csv:
        # Check if assignments is empty
        if not assignments:
            print("No assignments were made. Check your CSV format and try again.")
            return assignments

        # Find the maximum number of students assigned to any class
        max_students = max(len(names) for names in assignments.values())

        # Create a dictionary to store class data
        csv_data = {}

        # Group classes by time slot for better organization
        for slot in sorted(time_slots, key=time_slot_sort_key):
            slot_classes = [col for col in assignments.keys() if slot in col]

            for class_name in sorted(slot_classes):
                # Extract just the class number for the column header
                class_info = class_name.split("[")[1].strip("]")
                column_name = f"{slot} - {class_info}"

                # Add names for this class, padding with empty strings if needed
                names = assignments[class_name]
                csv_data[column_name] = names + [""] * (max_students - len(names))

        # Convert to DataFrame and save to CSV
        output_df = pd.DataFrame(csv_data)
        output_df.to_csv(output_csv, index=False)
        print(f"\nAssignments saved to {output_csv}")

    return assignments


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print(
            "Usage: python class_assignment.py <csv_file_path> <class_capacity> [output_csv]"
        )
        sys.exit(1)

    csv_file_path = sys.argv[1]
    class_capacity = int(sys.argv[2])
    output_csv = sys.argv[3] if len(sys.argv) > 3 else None

    process_csv_file(csv_file_path, class_capacity, output_csv)
