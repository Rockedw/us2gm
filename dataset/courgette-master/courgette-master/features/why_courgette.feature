Feature: Why courgette?

  In order to discuss the features of the application with the developpers
  As a Product Owner understanding the value of user acceptance criterias written with cucumber
  I want to have a simple read access to the latest version of cucumber features for my application (available on /features)

  Scenario: Courgette homepage
    Given I enter the address of the features page, e.g. http://staging.myfantasticapp.com/features
    Then I should see the source code of the first feature of my application

  Scenario: Not available in production env (I don't know how to test this)
