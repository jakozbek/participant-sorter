#!/usr/bin/env python3

import pandas as pd
import json
import heapq
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


def extract_class_name_from_column(col):
    """
    Extract the class name from a column header.

    Args:
        col (str): Column header like "Saturday, 9:00am-11:30am [Class Name (Instructor)]"

    Returns:
        str: The class name without instructor info
    """
    if "[" in col and "]" in col:
        class_info = col.split("[")[1].strip("]")
        if "(" in class_info and ")" in class_info:
            paren_start = class_info.rfind("(")
            return class_info[:paren_start].strip()
        return class_info.strip()
    return ""


def check_already_assigned_to_class(
    participant_name, class_name, participant_classes_assigned
):
    """
    Check if participant is already assigned to this class at another time.

    Args:
        participant_name (str): Name of the participant
        class_name (str): Name of the class to check
        participant_classes_assigned (dict): Dict tracking which classes each participant has been assigned to

    Returns:
        bool: True if already assigned to this class, False otherwise
    """
    return class_name in participant_classes_assigned.get(participant_name, set())


def check_prerequisites(
    participant_name, class_name, config, participant_classes_assigned
):
    """
    Check if participant has completed prerequisites for a class.

    Args:
        participant_name (str): Name of the participant
        class_name (str): Name of the class to check prerequisites for
        config (dict): Configuration dictionary
        participant_classes_assigned (dict): Dict of classes assigned to each participant

    Returns:
        bool: True if prerequisites are met, False otherwise
    """
    if "classes" not in config or class_name not in config["classes"]:
        return True

    required_classes = config["classes"][class_name].get("required_classes", [])
    if not required_classes:
        return True

    # Get all classes the participant is currently assigned to
    participant_classes = participant_classes_assigned.get(participant_name, set())

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


def participant_has_remaining_valid_preferences(
    participant_name,
    participant_preferences,
    user_assignments,
    participant_classes_assigned,
    class_columns,
    config,
    assignments,  # Add assignments parameter
):
    """
    Check if a participant has any remaining valid preferences they could potentially get.

    Args:
        participant_name (str): Name of the participant
        participant_preferences (dict): All preferences for this participant
        user_assignments (dict): Current time slot assignments
        participant_classes_assigned (dict): Classes already assigned
        class_columns (list): All class columns
        config (dict): Configuration dictionary
        assignments (dict): Current class assignments

    Returns:
        bool: True if they have remaining valid preferences
    """
    # Get available time slots for this participant
    time_slots = list(set(col.split("[")[0].strip() for col in class_columns))
    available_slots = []
    for slot in time_slots:
        if check_time_slot_conflicts(participant_name, slot, user_assignments, config):
            available_slots.append(slot)

    if not available_slots:
        return False

    # Check if they have any preferences in their available slots
    choice_levels = [
        "1st choice",
        "2nd choice",
        "3rd choice",
        "4th choice",
        "5th choice",
    ]

    for slot in available_slots:
        slot_classes = [col for col in class_columns if col.startswith(slot + " [")]
        for choice_level in choice_levels:
            for col in participant_preferences[participant_name][choice_level]:
                if col in slot_classes:
                    # Check they don't already have this class
                    class_name = extract_class_name_from_column(col)
                    if not check_already_assigned_to_class(
                        participant_name, class_name, participant_classes_assigned
                    ):
                        # Check prerequisites
                        if check_prerequisites(
                            participant_name,
                            class_name,
                            config,
                            participant_classes_assigned,
                        ):
                            # Check capacity constraint
                            class_capacity = get_class_capacity(class_name, config)
                            if len(assignments[col]) < class_capacity:
                                return True

    return False


def assign_participants_with_config(df, config):
    """
    Assign participants to classes based on their preferences, class capacities, and prerequisites.
    Uses a global priority queue where the person with fewest points always gets next pick.

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
    print(f"Identified {len(class_columns)} class columns")

    time_slots = list(set(col.split("[")[0].strip() for col in class_columns))
    print(f"Identified time slots: {time_slots}")

    # Initialize assignments dictionary
    assignments = {col: [] for col in class_columns}

    # Track which time slots each user is assigned to
    user_assignments = defaultdict(set)

    # Track which unique classes each user has been assigned to (regardless of time slot)
    participant_classes_assigned = defaultdict(set)

    # Track points for each user (lower points = higher priority)
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

    # Create participant preference mapping - now handling up to 5 choices
    participant_preferences = {}
    choice_levels = [
        "1st choice",
        "2nd choice",
        "3rd choice",
        "4th choice",
        "5th choice",
    ]

    for _, row in df.iterrows():
        full_name = row[name_col]
        preferences = {level: [] for level in choice_levels}

        for col in class_columns:
            if row[col] in preferences:
                preferences[row[col]].append(col)

        participant_preferences[full_name] = preferences

    # Updated points system: 5 for 1st, 4 for 2nd, 3 for 3rd, 2 for 4th, 1 for 5th
    choice_points_map = {
        "1st choice": 5,
        "2nd choice": 4,
        "3rd choice": 3,
        "4th choice": 2,
        "5th choice": 1,
    }

    # Create global priority queue: (points, timestamp, participant_name)
    priority_queue = []
    for _, row in df.iterrows():
        full_name = row[name_col]
        timestamp = timestamps.get(full_name, "")
        heapq.heappush(priority_queue, (0, timestamp, full_name))

    # Track which participants are fully processed (no more possible assignments)
    fully_processed = set()

    # Track attempts without any assignment to know when to stop
    attempts_without_assignment = 0
    max_attempts_per_point_level = len(df) * 4  # Max attempts at current point level

    # Track current point level being processed
    current_point_level = 0
    participants_at_current_level = set()

    print("\nStarting global assignment process...")
    assignments_made = 0

    # Process assignments until queue is empty or no more assignments possible
    while priority_queue and attempts_without_assignment < max_attempts_per_point_level:
        # Get the highest priority participant (lowest points, earliest timestamp)
        current_points, timestamp, participant_name = heapq.heappop(priority_queue)

        # Skip if we've determined this participant can't get any more assignments
        if participant_name in fully_processed:
            continue

        # Update current points (in case they've changed since being added to queue)
        current_points = user_points[participant_name]

        # Check if we've moved to a new point level
        if current_points != current_point_level:
            # We've advanced to a new point level
            if participants_at_current_level:
                print(
                    f"  Advanced from point level {current_point_level} to {current_points}"
                )
            current_point_level = current_points
            participants_at_current_level = set()
            attempts_without_assignment = 0  # Reset counter for new point level

        participants_at_current_level.add(participant_name)

        # Find the best available assignment for this participant
        assigned = False

        # Try each choice level in order
        for choice_level in choice_levels:
            if assigned:
                break

            choice_points = choice_points_map[choice_level]

            # Look through all their preferences at this level
            for col in participant_preferences[participant_name][choice_level]:
                # Extract time slot and class info
                slot = col.split("[")[0].strip()
                class_name = extract_class_name_from_column(col)

                # if participant_name == "theyearofplenty@gmail.com":
                print(
                    f"    🔍 Attempting: {participant_name} -> {class_name} ({choice_level}) in {slot}"
                )

                # Check all constraints with detailed logging
                class_capacity = get_class_capacity(class_name, config)

                # Check capacity constraint
                if len(assignments[col]) >= class_capacity:
                    print(
                        f"      ❌ BLOCKED: Class full ({len(assignments[col])}/{class_capacity})"
                    )
                    continue

                # Check time slot conflicts
                if not check_time_slot_conflicts(
                    participant_name, slot, user_assignments, config
                ):
                    conflicting_slots = get_conflicting_slots(slot, config)
                    assigned_slots = list(user_assignments[participant_name])
                    print(
                        f"      ❌ BLOCKED: Time conflict - slot '{slot}' conflicts with assigned slots {assigned_slots}"
                    )
                    continue

                # Check if already assigned to this class
                if check_already_assigned_to_class(
                    participant_name, class_name, participant_classes_assigned
                ):
                    print(f"      ❌ BLOCKED: Already assigned to '{class_name}'")
                    continue

                # Check prerequisites
                if not check_prerequisites(
                    participant_name, class_name, config, participant_classes_assigned
                ):
                    required_classes = (
                        config.get("classes", {})
                        .get(class_name, {})
                        .get("required_classes", [])
                    )
                    assigned_classes = list(
                        participant_classes_assigned[participant_name]
                    )
                    print(
                        f"      ❌ BLOCKED: Missing prerequisites - needs {required_classes}, has {assigned_classes}"
                    )
                    continue

                # All checks passed!
                print(f"      ✅ SUCCESS: All constraints satisfied")

                # Make the assignment
                assignments[col].append(participant_name)
                user_assignments[participant_name].add(slot)
                participant_classes_assigned[participant_name].add(class_name)

                # Award points
                points_multiplier = get_class_points_multiplier(class_name, config)
                user_points[participant_name] += choice_points * points_multiplier

                assigned = True
                assignments_made += 1
                attempts_without_assignment = 0

                print(
                    f"  [{assignments_made}] {participant_name} -> {class_name} in {slot} ({choice_level}, now at {user_points[participant_name]} points)"
                )
                break

        # Decide whether to re-add participant to queue
        if assigned:
            # Check if they have any remaining valid preferences
            if participant_has_remaining_valid_preferences(
                participant_name,
                participant_preferences,
                user_assignments,
                participant_classes_assigned,
                class_columns,
                config,
                assignments,
            ):
                # Re-add with updated points
                new_points = user_points[participant_name]
                heapq.heappush(
                    priority_queue, (new_points, timestamp, participant_name)
                )
            else:
                # No more possible assignments for this participant
                fully_processed.add(participant_name)
                print(
                    f"    -> {participant_name} fully processed (no remaining valid preferences)"
                )
        else:
            # They didn't get an assignment this round
            attempts_without_assignment += 1

            # Check if they still have potential assignments
            if participant_has_remaining_valid_preferences(
                participant_name,
                participant_preferences,
                user_assignments,
                participant_classes_assigned,
                class_columns,
                config,
                assignments,
            ):
                # They still have valid preferences but couldn't get them this round
                # Re-add to queue
                heapq.heappush(
                    priority_queue, (current_points, timestamp, participant_name)
                )
            else:
                # No more possible assignments
                fully_processed.add(participant_name)
                participants_at_current_level.discard(participant_name)

        # Check if we're stuck at the current point level
        if attempts_without_assignment >= max_attempts_per_point_level:
            print(f"\n🔍 DEBUGGING GRIDLOCK DETECTION at level {current_point_level}:")
            print(f"  attempts_without_assignment: {attempts_without_assignment}")
            print(f"  max_attempts_per_point_level: {max_attempts_per_point_level}")
            print(
                f"  participants_at_current_level: {len(participants_at_current_level)}"
            )

            # Print all participants at this leve
            for p in participants_at_current_level:
                print(f"    - {p} (Points: {user_points[p]})")

            # Check if there are participants at higher point levels who might still have assignments
            remaining_participants = []
            temp_queue = []

            # Extract all remaining participants from queue
            while priority_queue:
                item = heapq.heappop(priority_queue)
                temp_queue.append(item)
                if item[2] not in fully_processed:
                    remaining_participants.append(item)

            # Check if there are participants at higher point levels with valid preferences
            higher_point_participants = [
                p
                for p in remaining_participants
                if user_points[p[2]] > current_point_level
                and participant_has_remaining_valid_preferences(
                    p[2],
                    participant_preferences,
                    user_assignments,
                    participant_classes_assigned,
                    class_columns,
                    config,
                    assignments,
                )
            ]

            if higher_point_participants:
                print(
                    f"  Stuck at point level {current_point_level}, but found {len(higher_point_participants)} participants at higher levels with valid preferences"
                )
                print(
                    f"  Breaking gridlock by adding 1 point to all participants at level {current_point_level}"
                )

                # Add 1 point to all participants at the current gridlocked level
                gridlocked_participants = [
                    p
                    for p in remaining_participants
                    if user_points[p[2]] == current_point_level
                ]

                for participant_info in gridlocked_participants:
                    participant_name = participant_info[2]
                    user_points[participant_name] += 1
                    print(
                        f"    -> {participant_name}: {current_point_level} -> {user_points[participant_name]} points"
                    )

                # Rebuild the priority queue with updated points
                priority_queue.clear()
                for item in temp_queue:
                    participant_name = item[2]
                    if participant_name not in fully_processed:
                        # Use updated points for the priority
                        updated_points = user_points[participant_name]
                        heapq.heappush(
                            priority_queue, (updated_points, item[1], participant_name)
                        )

                # Reset counters
                attempts_without_assignment = 0
                participants_at_current_level.clear()
                # current_point_level will be updated naturally in the next iteration
            else:
                print(
                    f"  No participants at any point level have remaining valid preferences"
                )
                break

    if attempts_without_assignment >= max_attempts_per_point_level:
        print(f"\n⚠️  Stopped after being stuck at point level {current_point_level}")

    print(f"\nFully processed {len(fully_processed)} participants")

    # Print summary statistics
    print("\n" + "=" * 50)
    print("ASSIGNMENT SUMMARY")
    print("=" * 50)

    print(f"\nTotal assignments made: {assignments_made}")
    print(f"Total participants: {len(df)}")
    print(f"Participants with assignments: {len(user_assignments)}")
    print(f"Participants with no assignments: {len(df) - len(user_assignments)}")

    # Calculate assignment statistics
    assignment_counts = defaultdict(int)
    for participant, slots in user_assignments.items():
        assignment_counts[len(slots)] += 1

    print("\nAssignments per participant:")
    for count in sorted(assignment_counts.keys()):
        print(f"  {count} assignments: {assignment_counts[count]} participants")

    # Analyze missed opportunities
    print("\n" + "=" * 50)
    print("MISSED OPPORTUNITIES ANALYSIS")
    print("=" * 50)

    missed_opportunities = defaultdict(list)
    unfulfilled_by_reason = {
        "class_full": 0,
        "time_conflict": 0,
        "already_has_class": 0,
        "missing_prerequisite": 0,
        "no_preference": 0,
    }

    # For each participant, check what they could have gotten but didn't
    for _, row in df.iterrows():
        participant_name = row[name_col]

        # Check each time slot
        for slot in time_slots:
            # Skip if they're already assigned to this slot or a conflicting one
            if check_time_slot_conflicts(
                participant_name, slot, user_assignments, config
            ):
                # They could potentially be assigned to this slot
                slot_classes = [
                    col for col in class_columns if col.startswith(slot + " [")
                ]

                # Check their preferences for this slot
                eligible_but_missed = []

                for choice_level in choice_levels:
                    for col in participant_preferences[participant_name][choice_level]:
                        if col in slot_classes:
                            class_name = extract_class_name_from_column(col)

                            # Check why they didn't get this class
                            if col not in [
                                c
                                for c, participants in assignments.items()
                                if participant_name in participants
                            ]:
                                reason = ""

                                # Check various reasons for not getting it
                                if len(assignments[col]) >= get_class_capacity(
                                    class_name, config
                                ):
                                    reason = "class_full"
                                    unfulfilled_by_reason["class_full"] += 1
                                elif not check_time_slot_conflicts(
                                    participant_name, slot, user_assignments, config
                                ):
                                    reason = "time_conflict"
                                    unfulfilled_by_reason["time_conflict"] += 1
                                elif check_already_assigned_to_class(
                                    participant_name,
                                    class_name,
                                    participant_classes_assigned,
                                ):
                                    reason = "already_has_class"
                                    unfulfilled_by_reason["already_has_class"] += 1
                                elif not check_prerequisites(
                                    participant_name,
                                    class_name,
                                    config,
                                    participant_classes_assigned,
                                ):
                                    reason = "missing_prerequisite"
                                    unfulfilled_by_reason["missing_prerequisite"] += 1
                                else:
                                    reason = "available_but_not_assigned"

                                eligible_but_missed.append(
                                    {
                                        "class": class_name,
                                        "slot": slot,
                                        "choice_level": choice_level,
                                        "reason": reason,
                                    }
                                )

                if eligible_but_missed:
                    missed_opportunities[participant_name].extend(eligible_but_missed)

    # Report on participants with missed opportunities
    participants_with_missed = [
        p for p in missed_opportunities if missed_opportunities[p]
    ]

    print(f"\nParticipants with missed opportunities: {len(participants_with_missed)}")

    # Show details for participants who missed available slots
    available_but_not_assigned = []
    for participant_name, missed in missed_opportunities.items():
        available_missed = [
            m for m in missed if m["reason"] == "available_but_not_assigned"
        ]
        if available_missed:
            available_but_not_assigned.append((participant_name, available_missed))

    if available_but_not_assigned:
        print(
            "\n⚠️  WARNING: Some participants had available slots they wanted but didn't get:"
        )
        print("(This might indicate an algorithm issue or they were outprioritized)")
        for participant_name, missed in available_but_not_assigned[
            :10
        ]:  # Show first 10
            print(f"\n  {participant_name} (Points: {user_points[participant_name]}):")
            for m in missed[:3]:  # Show first 3 missed opportunities
                print(f"    - {m['class']} at {m['slot']} ({m['choice_level']})")

    # Analyze specific time slot utilization
    print("\n" + "=" * 50)
    print("TIME SLOT UTILIZATION ANALYSIS")
    print("=" * 50)

    slot_stats = defaultdict(lambda: {"capacity": 0, "assigned": 0, "classes": 0})

    for col in class_columns:
        slot = col.split("[")[0].strip()
        class_name = extract_class_name_from_column(col)
        capacity = get_class_capacity(class_name, config)
        assigned = len(assignments[col])

        slot_stats[slot]["capacity"] += capacity
        slot_stats[slot]["assigned"] += assigned
        slot_stats[slot]["classes"] += 1

    print("\nSlot utilization rates:")
    for slot in sorted(time_slots, key=time_slot_sort_key):
        stats = slot_stats[slot]
        utilization = (
            (stats["assigned"] / stats["capacity"] * 100)
            if stats["capacity"] > 0
            else 0
        )
        print(
            f"  {slot}: {stats['assigned']}/{stats['capacity']} seats filled ({utilization:.1f}%) across {stats['classes']} classes"
        )

    # Identify underutilized slots
    underutilized = []
    for slot, stats in slot_stats.items():
        utilization = (
            (stats["assigned"] / stats["capacity"] * 100)
            if stats["capacity"] > 0
            else 0
        )
        if utilization < 50:  # Less than 50% utilized
            underutilized.append((slot, utilization, stats))

    if underutilized:
        print("\n⚠️  Underutilized time slots (< 50% capacity):")
        for slot, utilization, stats in sorted(underutilized, key=lambda x: x[1]):
            print(
                f"  {slot}: only {utilization:.1f}% filled ({stats['assigned']}/{stats['capacity']} seats)"
            )

    # Print final points summary
    print("\n" + "=" * 50)
    print("Top 10 Participants by Priority Score:")
    print("=" * 40)
    sorted_participants = sorted(user_points.items(), key=lambda x: (-x[1], x[0]))[:10]
    for name, points in sorted_participants:
        slots_assigned = len(user_assignments.get(name, []))
        print(f"  {name}: {points} points ({slots_assigned} slots)")

    return assignments, user_points, timestamps, name_col


def create_debug_csv(
    df,
    user_points,
    timestamps,
    name_col,
    debug_csv_path,
    user_assignments,
    participant_classes_assigned,
):
    """
    Create an enhanced debug CSV file with detailed assignment information.

    Args:
        df (pandas.DataFrame): Original CSV data
        user_points (dict): Points awarded to each participant
        timestamps (dict): Timestamps for each participant
        name_col (str): Name of the participant name column
        debug_csv_path (str): Path for the debug CSV output
        user_assignments (dict): Time slots assigned to each participant
        participant_classes_assigned (dict): Classes assigned to each participant
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

        # Get assignment details
        num_assignments = len(user_assignments.get(participant_name, []))
        assigned_slots = ", ".join(sorted(user_assignments.get(participant_name, [])))
        assigned_classes = ", ".join(
            sorted(participant_classes_assigned.get(participant_name, []))
        )

        # Get first and last name if available
        first_name = row.get("First Name", "")
        last_name = row.get("Last Name", "")

        debug_data.append(
            {
                "Participant": participant_name,
                "First Name": first_name,
                "Last Name": last_name,
                "Points Awarded": points,
                "Number of Assignments": num_assignments,
                "Assigned Time Slots": assigned_slots if assigned_slots else "None",
                "Assigned Classes": assigned_classes if assigned_classes else "None",
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

    # Print unassigned participants
    unassigned = debug_df[debug_df["Number of Assignments"] == 0]
    if not unassigned.empty:
        print(f"\nParticipants with no assignments: {len(unassigned)}")
        for _, participant in unassigned.iterrows():
            print(f"  - {participant['Participant']}")


def time_slot_sort_key(slot):
    """Sort time slots chronologically"""
    # Extract time info from slots like "10 AM-12 PM", "1-3 PM", or "Saturday 9:00-10:15am"
    if "-" in slot:
        start_time = slot.split("-")[0].strip()

        # Check for day of week prefix
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
                day_prefix = (i + 1) * 100
                break

        # Handle hour formats
        if ":" in start_time:
            hour_str = "".join(c for c in start_time.split(":")[0] if c.isdigit())
            hour = int(hour_str) if hour_str else 0
        else:
            hour_str = "".join(c for c in start_time if c.isdigit())
            hour = int(hour_str) if hour_str else 0

        # Adjust for AM/PM
        if "pm" in start_time.lower() and hour < 12:
            hour += 12
        elif "am" in start_time.lower() and hour == 12:
            hour = 0

        return day_prefix + hour
    return 0


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
    print("Available columns in CSV:", df.columns.tolist()[:10], "...")

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
            if "csv_info" not in config:
                config["csv_info"] = {}
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

    # Recreate user_assignments and participant_classes_assigned for debug output
    user_assignments = defaultdict(set)
    participant_classes_assigned = defaultdict(set)

    for class_col, participants in assignments.items():
        slot = class_col.split("[")[0].strip()
        class_name = extract_class_name_from_column(class_col)

        for participant in participants:
            user_assignments[participant].add(slot)
            participant_classes_assigned[participant].add(class_name)

    # Create debug CSV
    debug_csv_path = csv_file_path.replace(".csv", "_debug.csv")
    create_debug_csv(
        df,
        user_points,
        timestamps,
        actual_name_col,
        debug_csv_path,
        user_assignments,
        participant_classes_assigned,
    )

    # Print results
    # print("\nClass Assignments:")
    # print("=================")
    #
    # Group by time slot for better readability
    time_slots = set(col.split("[")[0].strip() for col in assignments.keys())
    #
    # for slot in sorted(time_slots, key=time_slot_sort_key):
    #     print(f"\n{slot}:")
    #     slot_classes = [col for col in assignments.keys() if slot in col]
    #
    #     for class_name in sorted(slot_classes):
    #         class_info = class_name.split("[")[1].strip("]")
    #
    #         # Get class capacity
    #         extracted_class_name = extract_class_name_from_column(class_name)
    #         capacity = get_class_capacity(extracted_class_name, config)
    #
    #         print(f"  {class_info} ({len(assignments[class_name])}/{capacity}):")
    #         for participant_name in sorted(assignments[class_name]):
    #             print(f"    - {participant_name}")

    # Check assignment completeness
    participants = df[actual_name_col].tolist()
    participant_assignments = defaultdict(list)

    for class_name, names in assignments.items():
        time_slot = class_name.split("[")[0].strip()
        for participant_name in names:
            participant_assignments[participant_name].append(time_slot)

    print("\nParticipant Summary:")
    print("===================")

    # Count participants by number of slots assigned
    slot_counts = defaultdict(int)
    for participant_name in participants:
        assigned_slots = participant_assignments[participant_name]
        slot_counts[len(assigned_slots)] += 1

    for count in sorted(slot_counts.keys()):
        print(f"{slot_counts[count]} participants assigned to {count} time slots")

    # Create output CSV if requested
    if output_csv:
        # Check if assignments is empty
        if not assignments:
            print("No assignments were made. Check your CSV format and try again.")
            return assignments

        # Find the maximum number of students assigned to any class
        max_students = (
            max(len(names) for names in assignments.values()) if assignments else 0
        )

        # Create a dictionary to store class data
        csv_data = {}

        # Helper function to format names
        def format_name(full_name_or_email):
            if not use_short_names:
                return full_name_or_email

            # Find the row with this participant to get their first and last name
            if "First Name" in df.columns and "Last Name" in df.columns:
                matching_row = df[df[actual_name_col] == full_name_or_email]
                if not matching_row.empty:
                    first_name = matching_row.iloc[0]["First Name"]
                    last_name = matching_row.iloc[0]["Last Name"]
                    return f"{first_name} {last_name}"

            return full_name_or_email

        # Group classes by time slot for better organization
        for slot in sorted(time_slots, key=time_slot_sort_key):
            slot_classes = [
                col for col in assignments.keys() if col.startswith(slot + " [")
            ]

            for class_name in sorted(slot_classes):
                # Extract just the class number for the column header
                class_info = class_name.split("[")[1].strip("]")

                # Get class capacity for the header
                extracted_class_name = extract_class_name_from_column(class_name)
                capacity = get_class_capacity(extracted_class_name, config)

                # Create column name with enrollment/capacity
                current_enrollment = len(assignments[class_name])
                column_name = f"{slot} - {class_info} ({current_enrollment}/{capacity})"

                # Add names for this class, padding with empty strings if needed
                names = [format_name(name) for name in sorted(assignments[class_name])]
                csv_data[column_name] = names + [""] * (max_students - len(names))

        # Add summary column for unassigned participants
        unassigned = []
        for participant_name in participants:
            if not participant_assignments[participant_name]:
                unassigned.append(format_name(participant_name))

        if unassigned:
            csv_data["Unassigned Participants"] = unassigned + [""] * (
                max_students - len(unassigned)
            )

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
