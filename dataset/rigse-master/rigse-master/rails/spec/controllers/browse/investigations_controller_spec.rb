# frozen_string_literal: false

require 'spec_helper'

RSpec.describe Browse::InvestigationsController, type: :controller do

  # TODO: auto-generated
  describe '#show' do
    xit 'GET show' do
      get :show, params: { id: FactoryBot.create(:investigation).to_param }

      expect(response).to have_http_status(:ok)
    end
  end

end
