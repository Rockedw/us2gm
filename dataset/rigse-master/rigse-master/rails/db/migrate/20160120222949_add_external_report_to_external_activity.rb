class AddExternalReportToExternalActivity < ActiveRecord::Migration[5.1]
  def change
    add_column :external_activities, :external_report_id, :integer
  end
end
