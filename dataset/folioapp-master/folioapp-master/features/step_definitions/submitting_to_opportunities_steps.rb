Given(/^I have works$/) do
	visit '/'
  visit "/users/#{@user.id}"
  click_link 'Work'
  click_button 'Add To My Collection'
  click_link 'Add Art'
  attach_file 'Image', Rails.root.join('features/images/art.jpg')
  click_link 'Add Details'
  fill_in 'Title', with: 'Hello World'
  click_button 'Publish'

end

Given(/^I am on the profile of an organisation who have published an opportunity$/) do
	@organisation = Organisation.new name: "Notes", description: "Art, essays, writing", network: "Cambridge"
	@organisation.users << @user
	@organisation.save
	@organisation.opportunities.create title: "Issue 19", description: "Send us your stuff!"
	visit "/organisations/#{@organisation.id}"
end


Given(/^I click the 'Submit To Opportunity' link$/) do
	# save_and_open_page
	click_link 'Submit To Opportunity'
end

When(/^I select which work and add a message$/) do
	# save_and_open_page
	select 'My Collection: Hello World', from: 'Select Work'
	fill_in 'Message', with: "I hope you like my work :)"
end

When(/^I click the submit button$/) do
	click_button 'Submit'
end

When(/^I go on my homepage$/) do
	visit '/'
	# visit '/'
end

Then(/^I should see a thumbnail of my submission along with an acceptance status$/) do
	# save_and_open_page
	expect(page).to have_content 'Your Submissions'
	expect(page).to have_content 'Hello World'
	expect(page).to have_content 'Status: Pending'
	expect(page).to have_css 'img.submitted-pic'
end

