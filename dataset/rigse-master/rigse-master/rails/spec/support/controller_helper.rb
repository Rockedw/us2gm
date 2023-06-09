# This modification allows stubing helper methods when using integrate views
# the template object isn't ready until the render method is called, so this code
# adds a hook to be run before render is run.

# this commented out because it was breaking cucumber spork runs
# this ApplicationController definition was loaded before the main one so then
# the application controller wasn't extending ActionController::Base
# class ApplicationController
#   def before_render; end
#   def render(options=nil, extra_options={}, &bloc)
#     before_render
#     super
#   end
#
#   # any stub information is stored in the @mock_proxy variable of the object being stubbed,
#   # so adding it here prevents the controller @mock_proxy from clobbering the view @mock_proxy
#   # when rails copies the instance variables from the controller to view.  This copying happens
#   # sometime during the render method (after before_render)
#   @@protected_instance_variables = %w(@mock_proxy)
# end

#
# Factory Generators
#
def generate_default_settings_with_factories
  @admin_settings = FactoryBot.create(:admin_settings)
  generate_default_school_resources_with_factories
end

def generate_default_school_resources_with_factories
  @portal_school = FactoryBot.create(:portal_school)
  @portal_district = @portal_school.district
  @portal_grade_level = FactoryBot.create(:portal_grade_level)
  @portal_grade = @portal_grade_level.grade
end

#
# Mock Generators
#

# Generates mock settings
def generate_default_settings_with_mocks
  @mock_settings = mock_model(Admin::Settings,
    :active                         => true,
    :home_page_content              => nil,
    :use_student_security_questions => false,
    :use_bitmap_snapshots?          => false,
    :allow_adhoc_schools            => false,
    :allow_adhoc_schools?           => false,
    :require_user_consent?          => false,
    :allow_default_class            => false,
    :allow_default_class?           => false,
    :default_cohort                 => nil,
    :enabled_bookmark_types         => []
  )

  allow(Admin::Settings).to receive(:default_settings).and_return(@mock_settings)
  @mock_settings
end

def generate_portal_resources_with_mocks
  @mock_school = mock_model(Portal::School)
  @mock_grade = mock_model(Portal::Grade)
  @mock_grade_level = mock_model(Portal::GradeLevel)
  # @mock_grade ||= mock_model(Portal::Grade, stubs)
  # @mock_grade_level ||= mock_model(Portal::GradeLevel, stubs)
end

def login_admin
  logged_in_user = FactoryBot.generate :admin_user
  sign_in logged_in_user
  logged_in_user
end

def login_manager
  logged_in_user = FactoryBot.generate :manager_user
  sign_in logged_in_user
  logged_in_user
end

def login_researcher
  logged_in_user = FactoryBot.generate :researcher_user
  sign_in logged_in_user
  logged_in_user
end

def login_author
  logged_in_user = FactoryBot.generate :author_user
  sign_in logged_in_user
  logged_in_user
end

def login_anonymous
  logout_user
end

def logout_user
  sign_out :user
end

def will_paginate_params(opts = {})
  { :limit => opts[:limit] || 30, :offset => opts[:offset] || 0, :include=>opts[:include] || {} }
end
