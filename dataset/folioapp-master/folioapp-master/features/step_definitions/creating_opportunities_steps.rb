Given(/^I am a member of an organisation$/) do
	@organisation = Organisation.new name: "Notes", description: "Art, essays, writing", network: "Cambridge"
	@organisation.users << @user
	@organisation.save
end

Given(/^I am on my organisation profile$/) do
	visit "/organisations/#{@organisation.id}"
end

# Given(/^I click the 'Opportunities' link of my profile$/) do
# 	# save_and_open_page
# 	click_link 'Opportunities'
# end

Given(/^I specify that I wish to create an opportunity$/) do
	click_link 'Create Opportunity'
end

When(/^I give my opportunity a title, image, description and deadline, and requirements$/) do
	fill_in 'Title', with: 'Issue 19'
	attach_file 'opportunity_image', Rails.root.join('features/images/art.jpg')
	fill_in 'opportunity_description', with: 'Now accepting submissions of all kinds for the upcoming issue'
	# click_button '30' DEADLINE INFO GOES HERE
	select 'Both Image and Text', from: 'Requirements'
	click_button 'Submit'
end

Then(/^I should see my opportunity on my organisation page$/) do
	expect(page).to have_content 'Issue 19'
	expect(page).to have_css 'img.opportunity-image'
	expect(page).to have_content 'Now accepting submissions of all kinds for the upcoming issue'
	expect(page).to have_content 'Accepting both images and text'
	# deadline info
end