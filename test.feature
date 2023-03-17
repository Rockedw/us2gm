Feature: A user can signup and login to the web application

  As a user(student, mentor teacher, cal faculty)
  I want to register an account
  So that I can use the webapp



Scenario: create a student to database
  Given I am invited and on the signup page
  And  I fill in "First name" with "Sangyoon"
  And  I fill in "Last name" with "Park"
  And  I fill in "Street address" with "346 soda UC Berkeley"
  And  I fill in "City" with "Berkeley"
  And  I fill in "State" with "CA"
  And  I fill in "Zipcode" with "94000"
  And  I fill in "Phone number" with "123-456-7890"
  And  I fill in "Password" with "1234"
  And  I fill in "Password confirmation" with "1234"
  And  I press "Register"
  Then I should be located at "/users/1"
  And I should see "myemail@nowhere.com"

Scenario: I can't log in if I'm not registered
  Given I am signed up as a student advisor
  Given I am on the login page
  And I fill in "Email" with "wrong_email@email.com"
  And I fill in "Password" with "1234"
  And I press "Login"
  Then I should be located at "/user_sessions"
  And I should see "Email is not valid"
