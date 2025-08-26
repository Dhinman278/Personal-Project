import random
def number_guessing_game():
    
#welcome message
    print("welcome to the number guessing game")
    print("I'm thinking of a number between 1 and 100")

#generate a random number between 1 and 100
    secret_number = random.randint(1,100)
    attempts = 0
#loop until the user guesses the correct number
    while True:
        try:
            guess_string = input("Enter your guess")
            guess = int(guess_string)
            attempts += 1
            if guess <  secret_number:
                print("Too Low try again!")
            elif guess > secret_number:
                print("Too high try again!")
            else:
                print(f"congratulations you guessed the number {secret_number} in {attempts} attempts!")
                break
        except ValueError:
            print("Invalid input. Please enter a whole number.")
#start the game
if __name__ == "__main__":
    number_guessing_game()
