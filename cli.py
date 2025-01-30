#!/usr/bin/env python3

import os
import sys
import time
import atexit
import signal
import logging
import threading
import questionary
from datetime import datetime
from rich.table import Table
from rich.panel import Panel
from rich.console import Console
from rich.prompt import Prompt, Confirm
from typing import Dict, List, Optional
import requests
import json
from logging.handlers import RotatingFileHandler

from temp_mail_manager import (
    TempMailManager,
    GuerrillaMailAPI,
    DisposableMailAPI,
    YopMailAPI,
    TempMailOrgAPI,
    TempMailAPI
)

# Logging setup
def setup_logging():
    """Configure logging to write to files instead of console"""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup file handlers
    app_log = os.path.join(log_dir, 'app.log')
    error_log = os.path.join(log_dir, 'error.log')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    root_logger.handlers = []
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    # Setup file handlers
    file_handler = RotatingFileHandler(
        app_log,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    
    error_handler = RotatingFileHandler(
        error_log,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    
    # Prevent logs from being printed to console
    root_logger.propagate = False

# Initialize console
console = Console()

# Initialize manager
active_manager = TempMailManager()
setup_logging()

# Initialize stop events for monitoring
monitoring_stop_events = {}

# Global state
is_running = True
monitoring_threads = []

def display_goodbye():
    """Display goodbye message"""
    try:
        console.print(Panel(
            "[bold blue]Thank you for using TempMail Manager![/]\n"
            "[italic]Have a great day![/]",
            title="Goodbye!",
            border_style="green"
        ))
    except Exception:
        pass  # Ignore errors during cleanup

def cleanup(signum=None, frame=None):
    """Cleanup function to handle program exit"""
    try:
        # Save email data
        if active_manager:
            active_manager.save_data()
            
        # Stop all monitoring threads
        for thread, stop_event in monitoring_threads:
            stop_event.set()
            if thread.is_alive():
                thread.join(timeout=1)
                
        # Clear screen and display goodbye only if not already displayed
        if not hasattr(cleanup, 'goodbye_displayed'):
            display_goodbye()
            cleanup.goodbye_displayed = True
            
    except Exception as e:
        logging.getLogger('temp_mail_cli').error(f"Error during cleanup: {str(e)}")
        
    finally:
        # Ensure we exit
        if signum is not None:
            sys.exit(0)

def signal_handler(signum, frame):
    """Handle interrupt signals"""
    cleanup(signum, frame)

# Register cleanup handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(cleanup)

def list_emails(active_only=True) -> List[Dict]:
    """List all emails or only active ones"""
    try:
        emails = []
        if active_only:
            active_emails = active_manager.get_active_emails()
            for email in active_emails:
                provider = active_manager.get_provider_for_email(email)
                if provider:
                    emails.append({
                        'email': email,
                        'provider': provider.__class__.__name__,
                        'created_at': active_manager.emails[email].get('created_at', 'Unknown')
                    })
        else:
            for email, data in active_manager.emails.items():
                emails.append({
                    'email': email,
                    'provider': data.get('provider_class', 'Unknown'),
                    'created_at': data.get('created_at', 'Unknown')
                })
        return emails
    except Exception as e:
        logging.getLogger('temp_mail_cli').error(f"Error listing emails: {str(e)}")
        return []

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def display_welcome():
    """Display welcome message"""
    console.print(Panel(
        "[bold blue]Welcome to TempMail Manager![/]\n"
        "[italic]Your One-Stop Email Solution[/]",
        border_style="blue"
    ))

def display_menu():
    """Display main menu"""
    return questionary.select(
        "Choose an option:",
        choices=[
            "1. Generate new email",
            "2. Monitor emails",
            "3. List active emails",
            "4. Check messages",
            "5. Forward email",
            "6. Export emails",
            "7. Delete email",
            "8. Clear screen",
            "9. Exit"
        ],
        use_indicator=True
    ).ask()

def generate_email():
    """Generate a new email address"""
    try:
        # Let user select provider
        provider_choice = questionary.select(
            "Select email provider:",
            choices=[
                "Guerrilla Mail",
                "1secmail",
                "YopMail",
                "Temp-Mail.org"
            ]
        ).ask()
        
        if not provider_choice:
            return
            
        # Create provider instance
        provider = None
        if provider_choice == "Guerrilla Mail":
            provider = GuerrillaMailAPI()
        elif provider_choice == "1secmail":
            provider = DisposableMailAPI()
        elif provider_choice == "YopMail":
            provider = YopMailAPI()
        else:
            provider = TempMailOrgAPI()
            
        # Generate email
        email = provider.generate_email()
        if email:
            # Store provider instance in manager
            if active_manager.add_email(email, provider):
                console.print(f"\nGenerated new email: {email}")
                console.print(f"Provider: {provider.__class__.__name__}")
                
                # Verify provider is stored correctly
                stored_provider = active_manager.get_provider_for_email(email)
                if not stored_provider:
                    console.print("[red]Warning: Provider not stored correctly[/red]")
        else:
            console.print("[red]Failed to generate email[/red]")
            
    except Exception as e:
        logging.getLogger('temp_mail_cli').error(f"Error generating email: {str(e)}")
        console.print("[red]Failed to generate email. Please try another provider.[/red]")

def select_email_to_monitor():
    """Select an email to monitor from active emails"""
    try:
        if not active_manager.emails:
            console.print("[yellow]No active emails found. Generate an email first.[/]")
            return None

        table = Table(title="Active Email Addresses")
        table.add_column("Index", justify="center")
        table.add_column("Email Address", justify="left")
        table.add_column("Provider", justify="center")
        table.add_column("Created At", justify="center")

        for idx, email_data in enumerate(active_manager.emails.items(), 1):
            table.add_row(
                str(idx),
                email_data[0],
                email_data[1].get('provider_class', 'Unknown'),
                email_data[1].get('created_at', 'Unknown')
            )

        console.print(table)

        # Get user selection
        while True:
            choice = Prompt.ask(
                "Select email to monitor",
                default=1,
                show_default=True
            )
            
            if 1 <= choice <= len(active_manager.emails):
                selected_email = list(active_manager.emails.keys())[choice-1]
                console.print(f"\nSelected: {selected_email}")
                return selected_email
            else:
                console.print("[red]Invalid selection. Please try again.[/]")

    except Exception as e:
        logging.getLogger('temp_mail_cli').error(f"Error selecting email: {str(e)}")
        console.print("[red]Error selecting email. Please try again.[/]")
        return None

def display_message(message: dict):
    """Display a single message in a nice format"""
    panel = Panel(
        f"From: {message.get('from', 'Unknown')}\n"
        f"Subject: {message.get('subject', 'No Subject')}\n"
        f"Time: {datetime.fromtimestamp(int(message.get('date', 0))).strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Body: {message.get('body', '...')}",
        title="New Message",
        title_align="left"
    )
    console.print(panel)

def monitor_email_thread(email: str, stop_event: threading.Event):
    """
    Monitor email for new messages in a separate thread
    
    Args:
        email (str): Email address to monitor
        stop_event (threading.Event): Event to signal thread stop
    """
    try:
        last_check_time = time.time()
        check_interval = 10  # seconds
        timeout = 5  # seconds
        
        while not stop_event.is_set():
            try:
                current_time = time.time()
                if current_time - last_check_time >= check_interval:
                    provider = active_manager.get_provider_for_email(email)
                    if not provider:
                        logging.getLogger('temp_mail_cli').error(f"No provider found for email: {email}")
                        break
                        
                    # Get messages with timeout
                    messages = provider.get_messages(email)
                    if messages:
                        for message in messages:
                            display_message(message)
                            
                    last_check_time = current_time
                    
                # Sleep for a short interval to prevent CPU overuse
                time.sleep(1)
                
            except requests.Timeout:
                logging.getLogger('temp_mail_cli').warning(f"Timeout while checking messages for {email}")
                continue
            except Exception as e:
                logging.getLogger('temp_mail_cli').error(f"Error monitoring email {email}: {str(e)}")
                time.sleep(check_interval)  # Wait before retrying
                
    except Exception as e:
        logging.getLogger('temp_mail_cli').error(f"Monitor thread error for {email}: {str(e)}")
    finally:
        logging.getLogger('temp_mail_cli').info(f"Stopped monitoring {email}")

def start_monitoring(email: str):
    """Start monitoring an email address"""
    stop_event = threading.Event()
    thread = threading.Thread(target=monitor_email_thread, args=(email, stop_event))
    thread.daemon = True
    thread.start()
    monitoring_threads.append((thread, stop_event))
    console.print(f"[green]Started monitoring {email}[/green]")

def check_messages(email: str = None):
    """
    Check messages for an email address with timeout handling
    
    Args:
        email (str, optional): Email address to check. If None, user will be prompted
    """
    try:
        if not email:
            emails = list_emails(active_only=True)
            if not emails:
                console.print("[yellow]No active emails found. Generate an email first.[/]")
                return
                
            # Let user select email
            email_choice = questionary.select(
                "Select email to check:",
                choices=[e['email'] for e in emails]
            ).ask()
            
            if not email_choice:
                return
                
            email = email_choice
            
        # Get provider with timeout
        provider = active_manager.get_provider_for_email(email)
        if not provider:
            console.print(f"[red]No provider found for email: {email}[/]")
            return
            
        try:
            with console.status(f"[bold blue]Checking messages for {email}...[/]"):
                messages = active_manager.get_messages(email)
                
            if not messages:
                console.print("[yellow]No messages found.[/]")
                return
                
            # Display messages in a table
            table = Table(title=f"Messages for {email}")
            table.add_column("ID", justify="center")
            table.add_column("From", justify="left")
            table.add_column("Subject", justify="left")
            table.add_column("Date", justify="center")
            
            for msg in messages:
                table.add_row(
                    str(msg.get('id', 'N/A')),
                    msg.get('from', 'Unknown'),
                    msg.get('subject', 'No Subject'),
                    msg.get('date', 'Unknown')
                )
                
            console.print(table)
            
            # Ask if user wants to view a message
            if Confirm.ask("View a message?"):
                msg_id = Prompt.ask("Enter message ID")
                message = active_manager.get_message(email, msg_id)
                if message:
                    display_message(message)
                else:
                    console.print("[red]Message not found.[/]")
                    
        except requests.Timeout:
            console.print("[red]Request timed out. Please try again.[/]")
        except Exception as e:
            console.print(f"[red]Error checking messages: {str(e)}[/]")
            
    except Exception as e:
        logging.getLogger('temp_mail_cli').error(f"Error in check_messages: {str(e)}")
        console.print("[red]An error occurred while checking messages.[/]")

def monitor_emails():
    """Monitor emails for new messages"""
    try:
        if not active_manager.emails:
            console.print("[red]No active emails to monitor[/red]")
            return

        # Let user select emails to monitor
        emails = questionary.checkbox(
            "Select emails to monitor:",
            choices=list(active_manager.emails.keys())
        ).ask()

        if not emails:
            return

        for email in emails:
            start_monitoring(email)

        console.print("[green]Press Ctrl+C to stop monitoring[/green]")
        
        try:
            # Keep main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping email monitoring...[/yellow]")
            
    except Exception as e:
        logging.getLogger('temp_mail_cli').error(f"Error in monitor_emails: {str(e)}")
        console.print("[red]Error monitoring emails[/red]")

def stop_monitoring(email_address: str):
    """Stop monitoring an email address"""
    for thread, event in monitoring_threads[:]:
        event.set()
        thread.join(timeout=1)
        monitoring_threads.remove((thread, event))

def list_active_emails():
    """List all active emails"""
    try:
        emails = list_emails(active_only=True)
        if not emails:
            console.print("[yellow]No active emails found[/]")
            return
            
        table = Table(title="Active Emails")
        table.add_column("Email", style="cyan")
        table.add_column("Provider", style="green")
        table.add_column("Created At", style="blue")
        
        for email in emails:
            table.add_row(
                email['email'],
                email['provider'],
                email['created_at']
            )
            
        console.print(table)
        
    except Exception as e:
        logging.getLogger('temp_mail_cli').error(f"Error listing active emails: {str(e)}")
        console.print("[red]Failed to list active emails[/]")

def show_analytics():
    """Display email analytics"""
    try:
        if not active_manager or not active_manager.emails:
            console.print("[yellow]No email data available for analytics.[/]")
            return

        total_emails = len(active_manager.emails)
        active_emails = len([e for e in active_manager.emails if e.get('is_active', True)])
        
        table = Table(title="Email Analytics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Emails Generated", str(total_emails))
        table.add_row("Active Emails", str(active_emails))
        
        console.print(table)
        input("\nPress Enter to continue...")
    except Exception as e:
        logging.getLogger('temp_mail_cli').error(f"Error showing analytics: {str(e)}")
        console.print("[red]Failed to show analytics. Check logs for details.[/]")

def export_data():
    """Export email data"""
    try:
        format_choice = questionary.select(
            "Select export format:",
            choices=["json", "csv"]
        ).ask()
        
        if not format_choice:
            return
            
        # Export data using manager
        export_file = active_manager.export_data(format=format_choice)
        
        if export_file:
            console.print(f"[green]Data exported successfully to: {export_file}[/]")
        else:
            console.print("[red]Failed to export data[/]")
            
    except Exception as e:
        logging.getLogger('temp_mail_cli').error(f"Error exporting data: {str(e)}")
        console.print("[red]Failed to export data. Check logs for details.[/]")

def generate_test_data():
    """Generate test data"""
    try:
        if not active_manager:
            console.print("[red]TempMail Manager not initialized.[/]")
            return
            
        num_emails = questionary.text(
            "How many test emails would you like to generate?",
            validate=lambda text: text.isdigit() and int(text) > 0,
            default="5"
        ).ask()
        
        for _ in range(int(num_emails)):
            email = f"test_{_}@example.com"
            active_manager.emails[email] = {
                'provider_class': 'Test Provider',
                'created_at': datetime.now().isoformat(),
                'is_active': True
            }
        
        active_manager.save_data()
        console.print(f"[green]Successfully generated {num_emails} test emails[/]")
        input("\nPress Enter to continue...")
    except Exception as e:
        logging.getLogger('temp_mail_cli').error(f"Error generating test data: {str(e)}")
        console.print("[red]Failed to generate test data. Check logs for details.[/]")

def forward_email_menu():
    """Handle email forwarding menu with input validation"""
    try:
        # Get list of active emails
        emails = list_emails(active_only=True)
        if not emails:
            console.print("[yellow]No active emails found. Generate at least two emails first.[/]")
            return
            
        if len(emails) < 2:
            console.print("[yellow]You need at least two active emails to forward messages.[/]")
            return
            
        # Select source email
        from_email = questionary.select(
            "Select source email:",
            choices=[e['email'] for e in emails]
        ).ask()
        
        if not from_email:
            return
            
        # Get messages for source email
        try:
            messages = active_manager.get_messages(from_email)
            if not messages:
                console.print("[yellow]No messages found in source email.[/]")
                return
                
            # Select message to forward
            message_choices = [
                f"{msg.get('id', 'N/A')} - {msg.get('subject', 'No Subject')}"
                for msg in messages
            ]
            
            message_choice = questionary.select(
                "Select message to forward:",
                choices=message_choices
            ).ask()
            
            if not message_choice:
                return
                
            message_id = message_choice.split(' - ')[0]
            
            # Select target email
            to_email = questionary.select(
                "Select target email:",
                choices=[e['email'] for e in emails if e['email'] != from_email]
            ).ask()
            
            if not to_email:
                return
                
            # Confirm action
            if not Confirm.ask(f"Forward message from {from_email} to {to_email}?"):
                return
                
            # Forward message
            try:
                with console.status("[bold blue]Forwarding message...[/]"):
                    success = active_manager.forward_email(from_email, to_email, message_id)
                    
                if success:
                    console.print("[green]Message forwarded successfully![/]")
                else:
                    console.print("[red]Failed to forward message.[/]")
                    
            except Exception as e:
                console.print(f"[red]Error forwarding message: {str(e)}[/]")
                
        except Exception as e:
            console.print(f"[red]Error getting messages: {str(e)}[/]")
            return
            
    except Exception as e:
        logging.getLogger('temp_mail_cli').error(f"Error in forward_email_menu: {str(e)}")
        console.print("[red]An error occurred in the forwarding menu.[/]")

def delete_email():
    """Delete one or more emails with proper cleanup"""
    try:
        # Get list of active emails
        emails = list_emails(active_only=True)
        if not emails:
            console.print("[yellow]No active emails found.[/]")
            return
            
        # Ask user what to delete
        delete_option = questionary.select(
            "What would you like to delete?",
            choices=[
                "Single email",
                "Multiple emails",
                "All emails",
                "Cancel"
            ]
        ).ask()
        
        if not delete_option or delete_option == "Cancel":
            return
            
        emails_to_delete = []
        
        if delete_option == "Single email":
            # Select single email
            email_choice = questionary.select(
                "Select email to delete:",
                choices=[e['email'] for e in emails]
            ).ask()
            
            if email_choice:
                emails_to_delete.append(email_choice)
                
        elif delete_option == "Multiple emails":
            # Select multiple emails
            email_choices = questionary.checkbox(
                "Select emails to delete:",
                choices=[e['email'] for e in emails]
            ).ask()
            
            if email_choices:
                emails_to_delete.extend(email_choices)
                
        elif delete_option == "All emails":
            # Confirm deletion of all emails
            if Confirm.ask("Are you sure you want to delete all emails?"):
                emails_to_delete = [e['email'] for e in emails]
                
        # Perform deletion with cleanup
        if emails_to_delete:
            try:
                with console.status("[bold blue]Deleting emails...[/]"):
                    for email in emails_to_delete:
                        # Stop monitoring if active
                        stop_monitoring(email)
                        
                        # Delete from provider
                        provider = active_manager.get_provider_for_email(email)
                        if provider:
                            try:
                                provider.cleanup(email)
                            except Exception as e:
                                logging.getLogger('temp_mail_cli').warning(f"Provider cleanup failed for {email}: {str(e)}")
                                
                        # Delete from manager
                        active_manager.delete_email(email)
                        
                console.print(f"[green]Successfully deleted {len(emails_to_delete)} email(s)![/]")
                
            except Exception as e:
                console.print(f"[red]Error during deletion: {str(e)}[/]")
                
    except Exception as e:
        logging.getLogger('temp_mail_cli').error(f"Error in delete_email: {str(e)}")
        console.print("[red]An error occurred while deleting emails.[/]")

def handle_menu_option(option: str):
    """Handle menu option selection"""
    try:
        if option == "1. Generate new email":
            generate_email()
        elif option == "2. Monitor emails":
            monitor_emails()
        elif option == "3. List active emails":
            list_active_emails()
        elif option == "4. Check messages":
            check_messages()
        elif option == "5. Forward email":
            forward_email_menu()
        elif option == "6. Export emails":
            export_data()
        elif option == "7. Delete email":
            delete_email()
        elif option == "8. Clear screen":
            clear_screen()
            display_welcome()
        elif option == "9. Exit":
            return True  # Signal to exit
    except Exception as e:
        logging.getLogger('temp_mail_cli').error(f"Error handling option {option}: {str(e)}")
        console.print(f"[red]Error:[/] {str(e)}")
    return False

def main():
    """Main application entry point"""
    try:
        # Initialize manager and start CLI
        global active_manager
        active_manager = TempMailManager()
        
        # Clear screen and show welcome message
        clear_screen()
        display_welcome()
        
        while True:
            try:
                # Display menu and get user choice
                option = display_menu()
                
                if not option:  # User pressed Ctrl+C or similar
                    break
                    
                # Handle the selected option
                if handle_menu_option(option) or option == "9. Exit":
                    break
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logging.getLogger('temp_mail_cli').error(f"Error in main loop: {str(e)}")
                console.print("[red]An error occurred. Please try again.[/red]")
                
    except Exception as e:
        logging.getLogger('temp_mail_cli').error(f"Fatal error: {str(e)}")
    finally:
        cleanup()  # This will handle the goodbye message

if __name__ == "__main__":
    main()
