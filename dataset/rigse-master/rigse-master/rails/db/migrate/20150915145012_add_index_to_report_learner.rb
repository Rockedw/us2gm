class AddIndexToReportLearner < ActiveRecord::Migration[5.1]
  def change
  	add_index :report_learners, :student_id, name: 'index_report_learners_on_student_id'
  end
end
