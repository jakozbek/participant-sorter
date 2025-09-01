#!/usr/bin/env python3

import pandas as pd
import json
from collections import defaultdict


def load_config(config_path):
    """
    Load configuration from JSON file.

    Args:
        config_path (str): Path to the configuration JSON file

    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, "r") as f:
        return json.load(f)


def get_class_capacity(class_name, config, default_capacity=20):
    """
    Get capacity for a specific class from config.

    Args:
        class_name (str): Name of the class
        config (dict): Configuration dictionary
        default_capacity (int): Default capacity if not found in config

    Returns:
        int: Class capacity
    """
    if "classes" in config and class_name in config["classes"]:
        return config["classes"][class_name].get("capacity", default_capacity)
    return default_capacity


def get_class_points_multiplier(class_name, config, default_multiplier=1):
    """
    Get points multiplier for a specific class from config.

    Args:
        class_name (str): Name of the class
        config (dict): Configuration dictionary
        default_multiplier (int): Default multiplier if not found in config

    Returns:
        int: Points multiplier for the class
    """
    if "classes" in config and class_name in config["classes"]:
        return config["classes"][class_name].get(
            "points_multiplier", default_multiplier
        )
    return default_multiplier


def check_prerequisites(
    participant_name, class_name, config, user_assignments, class_columns
):
    """
    Check if participant has completed prerequisites for a class.

    Args:
        participant_name (str): Name of the participant
        class_name (str): Name of the class to check prerequisites for
        config (dict): Configuration dictionary
        user_assignments (dict): Current user assignments by time slot
        class_columns (list): List of all class columns

    Returns:
        bool: True if prerequisites are met, False otherwise
    """
    if "classes" not in config or class_name not in config["classes"]:
        return True

    required_classes = config["classes"][class_name].get("required_classes", [])
    if not required_classes:
        return True

    # Get all classes the participant is currently assigned to
    participant_classes = set()
    for assigned_slot in user_assignments[participant_name]:
        # Find which classes are in this assigned slot
        for col in class_columns:
            slot_from_col = col.split("[")[0].strip()
            if slot_from_col == assigned_slot:
                # Extract class name from column
                class_info = col.split("[")[1].strip("]")
                if "(" in class_info and ")" in class_info:
                    paren_start = class_info.rfind("(")
                    extracted_class_name = class_info[:paren_start].strip()
                else:
                    extracted_class_name = class_info.strip()
                participant_classes.add(extracted_class_name)

    # Check if any required class is satisfied
    for required_class in required_classes:
        if required_class in participant_classes:
            return True

    return False


def get_conflicting_slots(slot, config):
    """
    Get all time slots that conflict with the given slot.

    Args:
        slot (str): Time slot to check for conflicts
        config (dict): Configuration dictionary

    Returns:
        set: Set of conflicting time slots (including the original slot)
    """
    conflicting_slots = {slot}  # A slot always conflicts with itself

    if "time_slot_conflicts" in config:
        # Check if this slot has any defined conflicts
        if slot in config["time_slot_conflicts"]:
            conflicting_slots.update(config["time_slot_conflicts"][slot])

        # Check if this slot is listed as a conflict for other slots
        for other_slot, conflicts in config["time_slot_conflicts"].items():
            if slot in conflicts:
                conflicting_slots.add(other_slot)
                conflicting_slots.update(conflicts)

    return conflicting_slots


def check_time_slot_conflicts(participant_name, target_slot, user_assignments, config):
    """
    Check if assigning a participant to a time slot would create conflicts.

    Args:
        participant_name (str): Name of the participant
        target_slot (str): Time slot to check
        user_assignments (dict): Current user assignments by time slot
        config (dict): Configuration dictionary

    Returns:
        bool: True if there are no conflicts, False if there are conflicts
    """
    conflicting_slots = get_conflicting_slots(target_slot, config)

    # Check if user is already assigned to any conflicting slot
    for assigned_slot in user_assignments[participant_name]:
        if assigned_slot in conflicting_slots:
            return False

    return True


def assign_participants_with_config(df, config):
    """
    Assign participants to classes based on their preferences, class capacities, and prerequisites.

    Args:
        df (pandas.DataFrame): DataFrame with participant preferences
        config (dict): Configuration dictionary with class information

    Returns:
        tuple: (assignments dict, user_points dict, timestamps dict, name_col)
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
    timestamps = {}
    timestamp_col = config.get("csv_info", {}).get("timestamp_column", "Timestamp")
    name_col = config.get("csv_info", {}).get(
        "participant_name_column", "What is your full name?"
    )

    if timestamp_col in df.columns:
        for _, row in df.iterrows():
            timestamps[row[name_col]] = row[timestamp_col]

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
                    full_name = row[name_col]
                    # Skip if user already assigned to this time slot or any conflicting slots
                    if not check_time_slot_conflicts(
                        full_name, slot, user_assignments, config
                    ):
                        continue

                    # Extract class name for prerequisite checking
                    class_info = col.split("[")[1].strip("]")
                    if "(" in class_info and ")" in class_info:
                        paren_start = class_info.rfind("(")
                        class_name = class_info[:paren_start].strip()
                    else:
                        class_name = class_info.strip()

                    # Check prerequisites
                    if not check_prerequisites(
                        full_name, class_name, config, user_assignments, class_columns
                    ):
                        continue

                    # Check if this is their choice for this class
                    if row[col] == choice:
                        class_seekers[col].append(full_name)

            # Then assign users, prioritizing those with fewer points (less satisfied so far)
            for col in slot_columns:
                # Get class capacity from config
                class_info = col.split("[")[1].strip("]")
                if "(" in class_info and ")" in class_info:
                    paren_start = class_info.rfind("(")
                    class_name = class_info[:paren_start].strip()
                else:
                    class_name = class_info.strip()

                class_capacity = get_class_capacity(class_name, config)

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
                    if len(
                        assignments[col]
                    ) < class_capacity and check_time_slot_conflicts(
                        full_name, slot, user_assignments, config
                    ):
                        assignments[col].append(full_name)
                        user_assignments[full_name].add(slot)

                        # Award points based on which choice they got and class multiplier
                        points_multiplier = get_class_points_multiplier(
                            class_name, config
                        )
                        user_points[full_name] += choice_points * points_multiplier

    # Handle any users not yet assigned by finding available spots
    for _, row in df.iterrows():
        full_name = row[name_col]
        for slot in sorted(time_slots):
            # Skip if user already has a class in this slot or any conflicting slots
            if not check_time_slot_conflicts(full_name, slot, user_assignments, config):
                continue

            # Find classes with availability in this slot
            available_classes = []
            for col in class_columns:
                if slot in col:
                    # Get class capacity from config
                    class_info = col.split("[")[1].strip("]")
                    if "(" in class_info and ")" in class_info:
                        paren_start = class_info.rfind("(")
                        class_name = class_info[:paren_start].strip()
                    else:
                        class_name = class_info.strip()

                    class_capacity = get_class_capacity(class_name, config)

                    # Check prerequisites and availability
                    if len(assignments[col]) < class_capacity and check_prerequisites(
                        full_name, class_name, config, user_assignments, class_columns
                    ):
                        available_classes.append(col)

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

    return assignments, user_points, timestamps, name_col


def create_debug_csv(df, user_points, timestamps, name_col, debug_csv_path):
    """
    Create a debug CSV file with participant points and duplicate detection.

    Args:
        df (pandas.DataFrame): Original CSV data
        user_points (dict): Points awarded to each participant
        timestamps (dict): Timestamps for each participant
        name_col (str): Name of the participant name column
        debug_csv_path (str): Path for the debug CSV output
    """
    debug_data = []

    # Track participants we've seen to detect duplicates
    seen_participants = {}

    for _, row in df.iterrows():
        participant_name = row[name_col]
        timestamp = timestamps.get(participant_name, "No timestamp")
        points = user_points.get(participant_name, 0)

        # Check for duplicate submissions
        is_duplicate = False
        duplicate_info = ""

        if participant_name in seen_participants:
            is_duplicate = True
            previous_timestamp = seen_participants[participant_name]
            duplicate_info = f"Previous submission: {previous_timestamp}"
        else:
            seen_participants[participant_name] = timestamp

        # Get first and last name if available
        first_name = row.get("First Name", "")
        last_name = row.get("Last Name", "")

        debug_data.append(
            {
                "Participant": participant_name,
                "First Name": first_name,
                "Last Name": last_name,
                "Points Awarded": points,
                "Submission Timestamp": timestamp,
                "Is Duplicate": "Yes" if is_duplicate else "No",
                "Duplicate Info": duplicate_info,
            }
        )

    # Create DataFrame and save
    debug_df = pd.DataFrame(debug_data)
    debug_df = debug_df.sort_values(["Points Awarded", "Submission Timestamp"])
    debug_df.to_csv(debug_csv_path, index=False)

    print(f"\nDebug information saved to: {debug_csv_path}")

    # Print duplicate summary
    duplicates = debug_df[debug_df["Is Duplicate"] == "Yes"]
    if not duplicates.empty:
        print(f"Found {len(duplicates)} duplicate submissions:")
        for _, dup in duplicates.iterrows():
            print(f"  - {dup['Participant']}")
    else:
        print("No duplicate submissions detected.")


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
        if "pm" in start_time.lower() and hour < 12:
            hour += 12  # 1pm becomes 13, etc.
        elif "am" in start_time.lower():
            if hour == 12:
                hour = 0  # 12am becomes 0

        print(day_prefix + hour)

        return day_prefix + hour
    return 0  # Default for any other format


def process_csv_file(csv_file_path, config_path, output_csv=None, use_short_names=True):
    """
    Process a CSV file of participant preferences and assign them to classes.

    Args:
        csv_file_path (str): Path to the CSV file with preferences
        config_path (str): Path to configuration JSON file
        output_csv (str, optional): Path to output CSV file for assignments
        use_short_names (bool, optional): Use first and last name columns instead of full name/email in output

    Returns:
        dict: Dictionary with class names as keys and lists of participant names as values
    """
    # Read CSV file
    df = pd.read_csv(csv_file_path)

    # Display available columns to help with debugging
    print("Available columns in CSV:", df.columns.tolist())

    # Load config
    config = load_config(config_path)
    name_column = config.get("csv_info", {}).get(
        "participant_name_column", "What is your full name?"
    )

    # Check if name column exists
    if name_column not in df.columns:
        # Try to find alternative name column
        name_candidates = [
            col for col in df.columns if "name" in col.lower() or "email" in col.lower()
        ]
        if name_candidates:
            name_column = name_candidates[0]
            print(f"Using '{name_column}' as name column")
            # Update config
            config["csv_info"]["participant_name_column"] = name_column
        else:
            print("Error: No name column found in CSV file.")
            print(
                "Please ensure your CSV has a column with 'name' or 'email' in the title"
            )
            return {}

    # Run assignment algorithm
    assignments, user_points, timestamps, actual_name_col = (
        assign_participants_with_config(df, config)
    )

    # Create debug CSV
    debug_csv_path = csv_file_path.replace(".csv", "_debug.csv")
    create_debug_csv(df, user_points, timestamps, actual_name_col, debug_csv_path)

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

            # Get class capacity
            extracted_class_name = class_info
            if "(" in class_info and ")" in class_info:
                paren_start = class_info.rfind("(")
                extracted_class_name = class_info[:paren_start].strip()
            capacity = get_class_capacity(extracted_class_name, config)

            print(f"  Class {class_info} ({len(assignments[class_name])}/{capacity}):")
            for participant_name in assignments[class_name]:
                print(f"    - {participant_name}")

    # Check if any participant wasn't assigned to all time slots
    name_col = actual_name_col

    # Check if any participant wasn't assigned to all time slots
    name_col = (
        config.get("csv_info", {}).get(
            "participant_name_column", "What is your full name?"
        )
        if config
        else name_column
    )
    # Check if any participant wasn't assigned to all time slots
    participants = df[actual_name_col].tolist()
    participant_assignments = defaultdict(list)

    for class_name, names in assignments.items():
        time_slot = class_name.split("[")[0].strip()
        for participant_name in names:
            participant_assignments[participant_name].append(time_slot)

    print("\nParticipant Summary:")
    print("===================")
    for participant_name in participants:
        assigned_slots = participant_assignments[participant_name]
        missing_slots = [slot for slot in time_slots if slot not in assigned_slots]

        if missing_slots:
            print(
                f"{participant_name}: Missing assignments for {', '.join(missing_slots)}"
            )

    # Create output CSV if requested
    if output_csv:
        # Check if assignments is empty
        if not assignments:
            print("No assignments were made. Check your CSV format and try again.")
            return assignments

        # Find the maximum number of students assigned to any class
        max_students = max(len(names) for names in assignments.values())

        # Also consider missing participants when calculating max length
        max_missing_per_slot = 0
        for slot in time_slots:
            missing_count = 0
            for participant_name in participants:
                assigned_slots = participant_assignments[participant_name]
                if slot not in assigned_slots:
                    missing_count += 1
            max_missing_per_slot = max(max_missing_per_slot, missing_count)

        # Use the larger of the two for consistent column length
        max_length = max(max_students, max_missing_per_slot)

        # Create a dictionary to store class data
        csv_data = {}

        # Helper function to format names
        def format_name(full_name_or_email):
            if not use_short_names:
                return full_name_or_email

            # Find the row with this participant to get their first and last name
            if "First Name" in df.columns and "Last Name" in df.columns:
                matching_row = df[df[name_col] == full_name_or_email]
                if not matching_row.empty:
                    first_name = matching_row.iloc[0]["First Name"]
                    last_name = matching_row.iloc[0]["Last Name"]
                    return f"{first_name} {last_name}"

            # Fallback to original name if columns not found
            return full_name_or_email

        # Group classes by time slot for better organization
        for slot in sorted(time_slots, key=time_slot_sort_key):
            slot_classes = [col for col in assignments.keys() if slot in col]

            for class_name in sorted(slot_classes):
                # Extract just the class number for the column header
                class_info = class_name.split("[")[1].strip("]")

                # Get class capacity for the header
                extracted_class_name = class_info
                if "(" in class_info and ")" in class_info:
                    paren_start = class_info.rfind("(")
                    extracted_class_name = class_info[:paren_start].strip()
                capacity = get_class_capacity(extracted_class_name, config)

                # Create column name with enrollment/capacity
                current_enrollment = len(assignments[class_name])
                column_name = f"{slot} - {class_info} ({current_enrollment}/{capacity})"

                # Add names for this class, padding with empty strings if needed
                names = [format_name(name) for name in assignments[class_name]]
                csv_data[column_name] = names + [""] * (max_length - len(names))

        # Add columns for participants missing assignments
        for slot in sorted(time_slots, key=time_slot_sort_key):
            missing_for_slot = []
            for participant_name in participants:
                assigned_slots = participant_assignments[participant_name]
                if slot not in assigned_slots:
                    missing_for_slot.append(format_name(participant_name))

            if missing_for_slot:
                column_name = f"Missing {slot}"
                # Pad to match the maximum length
                missing_for_slot += [""] * (max_length - len(missing_for_slot))
                csv_data[column_name] = missing_for_slot

        # Convert to DataFrame and save to CSV
        output_df = pd.DataFrame(csv_data)
        output_df.to_csv(output_csv, index=False)
        print(f"\nAssignments saved to {output_csv}")

    return assignments


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print(
            "Usage: python class_assignment.py <csv_file_path> <config_path> [output_csv] [--short-names]"
        )
        print("  csv_file_path: Path to CSV file with participant preferences")
        print("  config_path: Path to JSON configuration file")
        print("  output_csv: Path to output CSV file (optional)")
        print(
            "  --short-names: Use first and last name instead of full name/email in output (optional)"
        )
        sys.exit(1)

    csv_file_path = sys.argv[1]
    config_path = sys.argv[2]

    # Parse remaining arguments
    output_csv = None
    use_short_names = True

    for arg in sys.argv[3:]:
        if arg == "--short-names":
            use_short_names = True
        elif arg.endswith(".csv"):
            output_csv = arg

    process_csv_file(csv_file_path, config_path, output_csv, use_short_names)
