import pandas as pd
from collections import defaultdict, Counter


def assign_participants(df, class_capacity):
    """
    Assign participants to classes based on their preferences and class capacity.

    Args:
        df (pandas.DataFrame): DataFrame with participant preferences
        class_capacity (int): Maximum number of participants per class

    Returns:
        dict: Dictionary with class names as keys and lists of participant emails as values
    """
    # Extract class columns and time slots
    class_columns = [col for col in df.columns if "[Class" in col]
    time_slots = set(col.split("[")[0].strip() for col in class_columns)

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
            timestamps[row["Email Address"]] = row["Timestamp"]

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
                    email = row["Email Address"]
                    # Skip if user already assigned to this time slot
                    if slot in user_assignments[email]:
                        continue
                    # Check if this is their choice for this class
                    if row[col] == choice:
                        class_seekers[col].append(email)

            # Then assign users, prioritizing those with fewer points (less satisfied so far)
            for col in slot_columns:
                # Define a key function for sorting users
                def sort_key(email):
                    # Primary sort by points (lower is better)
                    # Secondary sort by timestamp if available (earlier is better)
                    if email in timestamps:
                        return (user_points[email], timestamps[email])
                    return (user_points[email], "")

                # Sort users by points and timestamp
                sorted_users = sorted(class_seekers[col], key=sort_key)

                # Assign users until capacity is reached
                for email in sorted_users:
                    if (
                        len(assignments[col]) < class_capacity
                        and slot not in user_assignments[email]
                    ):
                        assignments[col].append(email)
                        user_assignments[email].add(slot)

                        # Award points based on which choice they got
                        user_points[email] += choice_points

    # Handle any users not yet assigned by finding available spots
    for _, row in df.iterrows():
        email = row["Email Address"]
        for slot in sorted(time_slots):
            # Skip if user already has a class in this slot
            if slot in user_assignments[email]:
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
                assignments[available_classes[0]].append(email)
                user_assignments[email].add(slot)
                # No points for fallback assignments

    return assignments


def process_csv_file(csv_file_path, class_capacity, output_csv=None):
    """
    Process a CSV file of participant preferences and assign them to classes.

    Args:
        csv_file_path (str): Path to the CSV file with preferences
        class_capacity (int): Maximum number of participants per class
        output_csv (str, optional): Path to output CSV file for assignments

    Returns:
        dict: Dictionary with class names as keys and lists of participant emails as values
    """
    # Read CSV file
    df = pd.read_csv(csv_file_path)

    # Display available columns to help with debugging
    print("Available columns in CSV:", df.columns.tolist())

    # Check if 'Email Address' exists, otherwise look for similar column
    email_column = "Email Address"
    if email_column not in df.columns:
        # Try to find alternative email column
        email_candidates = [col for col in df.columns if "email" in col.lower()]
        if email_candidates:
            email_column = email_candidates[0]
            print(f"Using '{email_column}' as email column")
            # Rename the column for consistency with the rest of the code
            df = df.rename(columns={email_column: "Email Address"})
        else:
            print("Error: No email column found in CSV file.")
            print("Please ensure your CSV has a column named 'Email Address'")
            return {}

    # Run assignment algorithm
    assignments = assign_participants(df, class_capacity)

    # Print results
    print("\nClass Assignments:")
    print("=================")

    # Group by time slot for better readability
    time_slots = set(col.split("[")[0].strip() for col in assignments.keys())

    for slot in sorted(time_slots):
        print(f"\n{slot}:")
        slot_classes = [col for col in assignments.keys() if slot in col]

        for class_name in sorted(slot_classes):
            class_num = class_name.split("[Class")[1].strip("]")
            print(
                f"  Class {class_num} ({len(assignments[class_name])}/{class_capacity}):"
            )
            for email in assignments[class_name]:
                print(f"    - {email}")

    # Check if any participant wasn't assigned to all time slots
    participants = df["Email Address"].tolist()
    participant_assignments = defaultdict(list)

    for class_name, emails in assignments.items():
        time_slot = class_name.split("[")[0].strip()
        for email in emails:
            participant_assignments[email].append(time_slot)

    print("\nParticipant Summary:")
    print("===================")
    for email in participants:
        assigned_slots = participant_assignments[email]
        missing_slots = [slot for slot in time_slots if slot not in assigned_slots]

        if missing_slots:
            print(f"{email}: Missing assignments for {', '.join(missing_slots)}")
        else:
            print(f"{email}: Fully assigned")

    # Create output CSV if requested
    if output_csv:
        # Find the maximum number of students assigned to any class
        max_students = max(len(emails) for emails in assignments.values())

        # Create a dictionary to store class data
        csv_data = {}

        # Group classes by time slot for better organization
        for slot in sorted(time_slots):
            slot_classes = [col for col in assignments.keys() if slot in col]

            for class_name in sorted(slot_classes):
                # Extract just the class number for the column header
                class_num = class_name.split("[Class")[1].strip("]")
                column_name = f"{slot} - Class {class_num}"

                # Add emails for this class, padding with empty strings if needed
                emails = assignments[class_name]
                csv_data[column_name] = emails + [""] * (max_students - len(emails))

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
