class AddAllowCollaborationToExternalActivities < ActiveRecord::Migration[5.1]
  def change
    add_column :external_activities, :allow_collaboration, :boolean, default: false
  end
end
