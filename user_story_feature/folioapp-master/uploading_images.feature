Feature: uploading images
	In order to have a nice portfolio
	As an artist
	I want to upload images to a collection on my profile
	And specify title, image-grouping, genre, tags, and captions

	Background:
		Given I am a user
		And on my profile
		# And I click the 'Work' link on my profile cover
		And I specify that I wish to upload art


	@javascript
	Scenario: uploading the first of a category in the right format without specifying group
		When I attach an image
		And I give it a title, medium, genre and captions
		And I click submit
		Then I should see the image in the default group
		And I should see the image in on its collection page

	@javascript
	Scenario: not uploading anything
		When I fail to attach an image
		Then I should not be allowed to submit


	@javascript
	Scenario: not specifying a title
		When I fail to specify a title
		Then I should not be allowed to submit



