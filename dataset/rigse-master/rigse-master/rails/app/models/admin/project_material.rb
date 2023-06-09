class Admin::ProjectMaterial < ApplicationRecord
  include Changeable

  self.table_name = 'admin_project_materials'

  belongs_to :project
  belongs_to :material, polymorphic: true
end
