import pytest
from temp_mail_manager import TempMailManager

def test_temp_mail_manager_initialization():
    manager = TempMailManager()
    assert manager is not None

def test_email_generation():
    manager = TempMailManager()
    email = manager.generate_email()
    assert '@' in email
    assert '.' in email

def test_save_and_load_data():
    manager = TempMailManager()
    # Generate an email
    email = manager.generate_email()
    # Save data
    manager.save_data()
    # Create new manager and load data
    new_manager = TempMailManager()
    assert email in new_manager.emails
