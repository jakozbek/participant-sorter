import unittest
import pandas as pd
from class_assignment import assign_participants
import datetime


def generate_test_users(
    num_users, time_slots, classes_per_slot, preferences_config=None
):
    """
    Generate test data with multiple users for testing the class assignment algorithm

    Args:
        num_users (int): Number of users to generate
        time_slots (int): Number of time slots
        classes_per_slot (int): Number of classes per time slot
        preferences_config (list, optional): List of dicts specifying preferences.
                                           If None, generates random preferences.

    Returns:
        pandas.DataFrame: DataFrame with generated user data
    """
    # Initialize data structure
    data = {"Timestamp": [], "Email Address": []}

    # Create column headers for all time slots and classes
    for slot in range(1, time_slots + 1):
        for class_num in range(1, classes_per_slot + 1):
            col_name = f"Time slot {slot} [Class {class_num}]"
            data[col_name] = []

    # Generate base timestamp
    base_timestamp = datetime.datetime(2024, 9, 20, 10, 0, 0)

    # Generate data for each user
    for i in range(num_users):
        # Generate timestamp with 5-minute increments
        timestamp = base_timestamp + datetime.timedelta(minutes=5 * i)
        data["Timestamp"].append(timestamp.strftime("%m/%d/%Y %H:%M:%S"))

        # Generate email
        data["Email Address"].append(f"user{i+1}@test.com")

        # Fill preferences
        if preferences_config and i < len(preferences_config):
            # Use specified preferences
            user_prefs = preferences_config[i]
            for slot in range(1, time_slots + 1):
                for class_num in range(1, classes_per_slot + 1):
                    col = f"Time slot {slot} [Class {class_num}]"
                    pref_key = (slot, class_num)
                    data[col].append(user_prefs.get(pref_key, ""))
        else:
            # Generate simple preferences - each user has one choice per time slot
            for slot in range(1, time_slots + 1):
                for class_num in range(1, classes_per_slot + 1):
                    col = f"Time slot {slot} [Class {class_num}]"
                    # First class is 1st choice, second is 2nd choice, third is 3rd choice
                    if class_num == ((i % classes_per_slot) + 1):
                        data[col].append("1st choice")
                    elif class_num == (((i + 1) % classes_per_slot) + 1):
                        data[col].append("2nd choice")
                    elif class_num == (((i + 2) % classes_per_slot) + 1):
                        data[col].append("3rd choice")
                    else:
                        data[col].append("")

    return pd.DataFrame(data)


class TestClassAssignment(unittest.TestCase):
    def test_prioritize_users_who_missed_first_choice(self):
        """
        Test that users who didn't get their first choice are prioritized in subsequent rounds.

        Setup:
        - 6 users (A-F)
        - 2 time slots with 3 classes total (Morning[Class1], Morning[Class2], Afternoon[Class3])
        - 3 students per class limit

        Scenario:
        - Users A, B, C, D all want Morning[Class1] as first choice, but only 3 can get in
        - User A, B, C get Morning[Class1]
        - User D misses first choice and should be prioritized for their second choice
        - User D and E both want Afternoon[Class3] as second choice
        - User D should get priority over E because D missed their first choice
        """
        # Create test data
        data = {
            "Email Address": [
                "A@test.com",
                "B@test.com",
                "C@test.com",
                "D@test.com",
                "E@test.com",
                "F@test.com",
            ],
            "Morning [Class1]": [
                "1st choice",
                "1st choice",
                "1st choice",
                "1st choice",
                "3rd choice",
                "2nd choice",
            ],
            "Morning [Class2]": [
                "2nd choice",
                "2nd choice",
                "2nd choice",
                "3rd choice",
                "1st choice",
                "3rd choice",
            ],
            "Afternoon [Class3]": [
                "3rd choice",
                "3rd choice",
                "3rd choice",
                "2nd choice",
                "2nd choice",
                "1st choice",
            ],
        }

        df = pd.DataFrame(data)
        class_capacity = 3

        # Run the algorithm
        assignments = assign_participants(df, class_capacity)

        # Verify assignments
        # Check Class1 has exactly 3 people
        self.assertEqual(len(assignments["Morning [Class1]"]), 3)

        # Check all first 3 users got their first choice
        for email in ["A@test.com", "B@test.com", "C@test.com"]:
            self.assertIn(email, assignments["Morning [Class1]"])

        # Check D didn't get first choice
        self.assertNotIn("D@test.com", assignments["Morning [Class1]"])

        # Most importantly: check D got priority for second choice over E
        self.assertIn("D@test.com", assignments["Afternoon [Class3]"])

        # Check F got their first choice (Afternoon [Class3])
        self.assertIn("F@test.com", assignments["Afternoon [Class3]"])

    def test_many_users_competing_for_limited_spots(self):
        # Generate 20 users with 3 time slots and 5 classes each
        # Most users will want class 1 as first choice
        preferences = []

        # First 10 users all want Class 1 first, Class 2 second
        for i in range(10):
            preferences.append(
                {
                    (1, 1): "1st choice",
                    (1, 2): "2nd choice",
                    (1, 3): "3rd choice",
                    (2, 1): "1st choice",
                    (2, 2): "2nd choice",
                    (3, 4): "1st choice",
                    (3, 5): "2nd choice",
                }
            )

        # Next 10 users want different combinations
        for i in range(10):
            slot1_class = (i % 5) + 1
            slot2_class = ((i + 2) % 5) + 1
            preferences.append(
                {
                    (1, slot1_class): "1st choice",
                    (1, ((slot1_class) % 5) + 1): "2nd choice",
                    (2, slot2_class): "1st choice",
                    (3, ((i + 1) % 5) + 1): "1st choice",
                }
            )

        df = generate_test_users(
            20, time_slots=3, classes_per_slot=5, preferences_config=preferences
        )

        # Class size of 5
        assignments = assign_participants(df, 5)

        # Check that each class has no more than 5 participants
        for class_name, participants in assignments.items():
            self.assertLessEqual(len(participants), 5)

        # Check that everyone who didn't get their first choice in one slot
        # was prioritized for their first choice in other slots

        # Count assignments by user
        user_assignments = {}
        for class_name, participants in assignments.items():
            slot = int(class_name.split(" ")[2])
            for user in participants:
                if user not in user_assignments:
                    user_assignments[user] = {"slots": {}, "first_choices": 0}
                user_assignments[user]["slots"][slot] = class_name

        # Check how many users got at least one first choice
        for user_email in df["Email Address"]:
            user = df[df["Email Address"] == user_email].iloc[0]
            first_choice_count = 0

            if user_email not in user_assignments:
                continue  # User wasn't assigned any class

            # Count user's first choices that were fulfilled
            for slot in range(1, 4):  # 3 time slots
                for class_num in range(1, 6):  # 5 classes
                    col = f"Time slot {slot} [Class {class_num}]"
                    if col in df.columns and user[col] == "1st choice":
                        class_name = col
                        if (
                            slot in user_assignments[user_email]["slots"]
                            and user_assignments[user_email]["slots"][slot]
                            == class_name
                        ):
                            first_choice_count += 1

            user_assignments[user_email]["first_choices"] = first_choice_count

        # Count users who got at least one first choice
        users_with_first_choice = sum(
            1 for u in user_assignments.values() if u["first_choices"] > 0
        )

        # Assert that at least 80% of users got at least one of their first choices
        self.assertGreaterEqual(
            users_with_first_choice, int(0.8 * len(user_assignments))
        )

    def test_random_large_dataset(self):
        # Generate 100 users with 5 time slots and 8 classes each
        df = generate_test_users(100, time_slots=5, classes_per_slot=8)

        # Test with class size limit of 15
        assignments = assign_participants(df, 15)

        # Check that no class exceeds capacity
        for class_name, participants in assignments.items():
            self.assertLessEqual(len(participants), 15)

        # Check that all users are assigned to at most one class per time slot
        user_assignments = {}
        for class_name, participants in assignments.items():
            time_slot = class_name.split("[")[0].strip()
            for participant in participants:
                if participant not in user_assignments:
                    user_assignments[participant] = set()
                user_assignments[participant].add(time_slot)

        for user, slots in user_assignments.items():
            for slot in slots:
                # Count classes in this slot
                classes_in_slot = 0
                for class_name in assignments:
                    if slot in class_name and user in assignments[class_name]:
                        classes_in_slot += 1
                # User should be in at most one class per slot
                self.assertLessEqual(classes_in_slot, 1)


if __name__ == "__main__":
    unittest.main()
